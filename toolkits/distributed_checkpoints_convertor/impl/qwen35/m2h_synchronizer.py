# Copyright (c) 2025 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
import logging

import torch
from typing import Dict
from general.m2h_synchronizer import MG2HFSynchronizer as _MG2HFSynchronizer, ParamMergeError
from general.synchronizer import ParamType

class MG2HFSynchronizer(_MG2HFSynchronizer):
    # TODO: to be refactored to Hybrid Model convertor
    def __init__(self, load_dir, model_provider_func=None):
        super().__init__(load_dir, model_provider_func)
        self.layout = self.get_hybrid_layout()
        assert self.tp_size == 1, "Currently MCore2HF conversion for Qwen3.5 is only available with TP 1."

    def get_hybrid_layout(self) -> str:
        assert self.args.hybrid_override_pattern is not None
        return self.args.hybrid_override_pattern

    def get_export_state_dict(self):
        return {
            key: value
            for key, value in self._hfmodel.state_dict().items()
            if ".visual." not in key and not key.startswith("visual.")
        }

    def _copy_impl(self, src_tensor, dst_tensor, param_type: ParamType=ParamType.UNIQUE):
        if param_type == ParamType.AGGREGATION_MOE_ROW:
            param_id = self._hf_params_to_id[dst_tensor]
            if self.edp_rank != 0:
                return
            self._local_params[param_id] = src_tensor
            self._merge_type[param_id] = param_type.value
            return
        return super()._copy_impl(src_tensor, dst_tensor, param_type=param_type)

    def _merge_data(self, merge_type: ParamType, tensor_dict) -> torch.Tensor:
        if merge_type not in [ParamType.MOE_GATE_UP, ParamType.AGGREGATION_MOE_ROW]:
            return super()._merge_data(merge_type, tensor_dict)

        global_ranks = torch.tensor(list(tensor_dict.keys()), dtype=torch.long, device=self.device)
        ranks = self._rank_mapping.index_select(0, global_ranks)
        tensor_dict = {k: v for k, v in tensor_dict.items() if self._rank_mapping[k, 1] == ranks[:, 1].min()}

        if self.debug:
            filtered_ranks = self._rank_mapping.index_select(
                0,
                torch.tensor(list(tensor_dict.keys()), dtype=torch.long, device=self.device),
            )
            if filtered_ranks[:, 5].any():
                raise ParamMergeError("Unexpected expert parameter data from non-zero expert dp rank")

        tensors_by_ep = defaultdict(dict)
        for global_rank, data in tensor_dict.items():
            ep_rank = int(self._rank_mapping[global_rank][3])
            etp_rank = int(self._rank_mapping[global_rank][2])
            tensors_by_ep[ep_rank][etp_rank] = data

        merged = []
        for ep_rank in sorted(tensors_by_ep.keys()):
            merged.append(torch.cat([shard for _, shard in sorted(tensors_by_ep[ep_rank].items())], dim=2))

        result = torch.cat(merged, dim=0)
        if merge_type == ParamType.MOE_GATE_UP:
            return result.flatten(1, 2)
        return result

    def get_aggregation_mlp_state(self, mlp, expert_id=''):
        hidden_size = getattr(mlp.linear_fc1, f'weight{expert_id}').shape[-1]
        gate_up_proj_weight = getattr(mlp.linear_fc1, f'weight{expert_id}').reshape(2, -1, hidden_size)
        return gate_up_proj_weight, getattr(mlp.linear_fc2, f'weight{expert_id}')

    def set_sequential_aggregation_mlp_state(self, experts, hf_experts):
        experts = experts.local_experts
        gate_up_proj_weights = []
        down_proj_weights = []
        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            del hf_expert_id
            gate_up_proj_weight, down_proj_weight = self.get_aggregation_mlp_state(experts[mg_expert_id])
            gate_up_proj_weights.append(gate_up_proj_weight)
            down_proj_weights.append(down_proj_weight)

        self.copy(torch.stack(gate_up_proj_weights), hf_experts.gate_up_proj, param_type=ParamType.MOE_GATE_UP)
        self.copy(torch.stack(down_proj_weights), hf_experts.down_proj, param_type=ParamType.AGGREGATION_MOE_ROW)

    def set_group_aggregation_mlp_state(self, experts, hf_experts):
        gate_up_proj_weights = []
        down_proj_weights = []
        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            del hf_expert_id
            gate_up_proj_weight, down_proj_weight = self.get_aggregation_mlp_state(experts, expert_id=mg_expert_id)
            gate_up_proj_weights.append(gate_up_proj_weight)
            down_proj_weights.append(down_proj_weight)

        self.copy(torch.stack(gate_up_proj_weights), hf_experts.gate_up_proj, param_type=ParamType.MOE_GATE_UP)
        self.copy(torch.stack(down_proj_weights), hf_experts.down_proj, param_type=ParamType.AGGREGATION_MOE_ROW)

    def set_aggregation_moe_layer_state(self, moe, hf_moe):
        self.copy(moe.router.weight, hf_moe.gate.weight)
        if moe.router.enable_expert_bias:
            self.copy(moe.router.expert_bias, hf_moe.gate.e_score_correction_bias)

        if self.args.moe_grouped_gemm:
            if self.args.moe_use_legacy_grouped_gemm:
                raise NotImplementedError("Currently only TE GroupGEMM is implemented.")
            self.set_group_aggregation_mlp_state(moe.experts, hf_moe.experts)
        else:
            self.set_sequential_aggregation_mlp_state(moe.experts, hf_moe.experts)

        if moe.shared_experts is not None:
            if moe.shared_experts.use_shared_expert_gate:
                self.copy(moe.shared_experts.gate_weight, hf_moe.shared_expert_gate.weight)

            try:
                hf_shared_expert = hf_moe.shared_experts
            except AttributeError:
                hf_shared_expert = hf_moe.shared_expert
            self.set_mlp_state(moe.shared_experts, hf_shared_expert)


    def sync_params(self, mg_model = None, hf_model = None):
        # assume TE backend
        if self.args.transformer_impl != "transformer_engine":
            raise NotImplementedError("Currently only TE model is implemented.")
        
        if mg_model is None:
            mg_model = self._mgmodel
        if hf_model is None:
            hf_model = self._hfmodel

        if mg_model.pre_process:
            self.copy(
                mg_model.embedding.word_embeddings.weight,
                hf_model.model.language_model.embed_tokens.weight,
                param_type=ParamType.COLUMN
            )
            
        if mg_model.post_process:
            self.copy(
                mg_model.decoder.final_norm.weight,
                hf_model.model.language_model.norm.weight,
            )

            if mg_model.share_embeddings_and_output_weights:
                output_layer_weight = mg_model.shared_embedding_or_output_weight()
            else:
                output_layer_weight = mg_model.output_layer.weight
            self.copy(output_layer_weight, hf_model.lm_head.weight, param_type=ParamType.COLUMN)

            # mtp layer
            self.set_aggregation_moe_layer_state(mg_model.mtp.layers[0].transformer_layer.mlp, hf_model.mtp.layers[0].mlp)
            self.copy(mg_model.mtp.layers[0].transformer_layer.pre_mlp_layernorm.weight, hf_model.mtp.layers[0].post_attention_layernorm.weight)
            self.set_gated_selfattn_state(mg_model.mtp.layers[0].transformer_layer.self_attention, hf_model.mtp.layers[0].self_attn)
            self.copy(mg_model.mtp.layers[0].transformer_layer.self_attention.linear_qgkv.layer_norm_weight, hf_model.mtp.layers[0].input_layernorm.weight)

            self.copy(mg_model.mtp.layers[0].enorm.weight, hf_model.mtp.pre_fc_norm_embedding.weight)
            self.copy(mg_model.mtp.layers[0].hnorm.weight,hf_model.mtp.pre_fc_norm_hidden.weight)
            self.copy(mg_model.mtp.layers[0].final_layernorm.weight, hf_model.mtp.norm.weight)

            self.copy(
                mg_model.mtp.layers[0].eh_proj.weight,
                hf_model.mtp.fc.weight,
                param_type=ParamType.COLUMN
            )


        for mg_layer_id, global_mg_layer_id in self._build_pipeline_parallel_mapping().items():
            hf_layer_id  = global_mg_layer_id // 2
            if (
                self.tp_rank == 0 and 
                self.ep_rank == 0 and 
                self.etp_rank == 0 and
                global_mg_layer_id % 2 == 0
            ):
                logging.info(f"Converting layer {hf_layer_id}")

            layer = mg_model.decoder.layers[mg_layer_id]
            hf_layer = hf_model.model.language_model.layers[hf_layer_id]

            if self.layout[global_mg_layer_id] == 'M':
                # Mamba layer
                self.set_mamba_layer_state(layer.mixer, hf_layer.linear_attn)
                self.copy(layer.mixer.in_proj.layer_norm_weight, hf_layer.input_layernorm.weight)
            elif self.layout[global_mg_layer_id] == '-':
                # transformer_layer of MLP
                self.set_aggregation_moe_layer_state(layer.mlp, hf_layer.mlp)
                self.copy(layer.pre_mlp_layernorm.weight, hf_layer.post_attention_layernorm.weight)
            elif self.layout[global_mg_layer_id] == '*':
                # transformer_layer of Attention
                self.set_gated_selfattn_state(layer.self_attention, hf_layer.self_attn)
                self.copy(layer.self_attention.linear_qgkv.layer_norm_weight, hf_layer.input_layernorm.weight)
            else:
                raise ValueError(f"Unrecognized layer type {self.layout[global_mg_layer_id]} in {self.layout}")

    def set_mamba_layer_state(self, mixer, hf_mixer):
        Nk, Nv, Dk, Dv = (
            hf_mixer.num_k_heads,
            hf_mixer.num_v_heads,
            hf_mixer.head_k_dim,
            hf_mixer.head_v_dim
        )
        split_size_list = [
            Dv * Nv // self.tp_size, 
            Dv * Nv // self.tp_size, 
            Dk * Nk // self.tp_size, 
            Dk * Nk // self.tp_size, 
            Nv // self.tp_size, 
            Nv // self.tp_size
        ]
        z, v, q, k, b, a = torch.split(
            mixer.in_proj.weight,
            split_size_list,
            dim=0
        )
        q_proj_weight = q.reshape(Nk // self.tp_size, Dk, -1)
        k_proj_weight = k.reshape(Nk // self.tp_size, Dk, -1)
        v_proj_weight = v.reshape(Nk // self.tp_size, Dv * Nv // Nk, -1)
        z_proj_weight = z.reshape(Nk // self.tp_size, Dv * Nv // Nk, -1)
        b_proj_weight = b.reshape(Nk // self.tp_size, Nv // Nk, -1)
        a_proj_weight = a.reshape(Nk // self.tp_size, Nv // Nk, -1)

        self.copy(
            torch.cat(
                [
                    q_proj_weight.flatten(0, 1),
                    k_proj_weight.flatten(0, 1),
                    v_proj_weight.flatten(0, 1),
                ],
                dim=0,
            ),
            hf_mixer.in_proj_qkv.weight,
            param_type=ParamType.UNIQUE,
        )
        self.copy(z_proj_weight, hf_mixer.in_proj_z.weight, param_type=ParamType.QKV_W)
        self.copy(b_proj_weight, hf_mixer.in_proj_b.weight, param_type=ParamType.QKV_W)
        self.copy(a_proj_weight, hf_mixer.in_proj_a.weight, param_type=ParamType.QKV_W)

        self.copy(mixer.dt_bias, hf_mixer.dt_bias, param_type=ParamType.COLUMN)
        self.copy(mixer.A_log, hf_mixer.A_log, param_type=ParamType.COLUMN)

        split_size_list = [
            Nv * Dv, 
            Nk * Dk, 
            Nk * Dk, 
        ]
        conv_v, conv_q, conv_k = torch.split(
            mixer.conv1d.weight, 
            split_size_or_sections=split_size_list, 
            dim=0
        )
        conv1d_weight = torch.cat([
            conv_q, 
            conv_k, 
            conv_v, 
        ], dim=0)
        self.copy(conv1d_weight, hf_mixer.conv1d.weight, param_type=ParamType.UNIQUE)
        
        self.copy(mixer.norm.weight, hf_mixer.norm.weight, param_type=ParamType.UNIQUE)
        self.copy(mixer.out_proj.weight, hf_mixer.out_proj.weight, param_type=ParamType.ROW)

    def _build_pipeline_parallel_mapping(self) -> Dict[int, int]:
        remained_num_layers = self.args.num_layers
        remained_stages = self.pp_size
        pp_layers_per_stage = [remained_num_layers // remained_stages] * remained_stages

        pp_mapping = {
            i: v for i, v in enumerate(
                range(
                    sum(pp_layers_per_stage[:self.pp_rank]), 
                    sum(pp_layers_per_stage[:self.pp_rank + 1])
                )
            )
        }
        return pp_mapping

    def set_gated_selfattn_state(self, attn, hf_attn):
        '''Set gated self-attention params.'''
        # Reshape loaded weights.
        tp = self.tp_size
        num_heads = self.args.num_attention_heads
        num_query_groups = (self.args.num_query_groups if self.args.group_query_attention else self.args.num_attention_heads)
        num_querys_per_group = num_heads // num_query_groups
        dim = self.args.kv_channels
        assert num_heads % num_querys_per_group == 0
        # copy qk norm if indeed.
        if self.args.qk_layernorm:
            self.copy(attn.q_layernorm.weight, hf_attn.q_norm.weight)
            self.copy(attn.k_layernorm.weight, hf_attn.k_norm.weight)

        # Copy weights (re-order dimensions for Megatron).
        attn_proj_weight = attn.linear_qgkv.weight.reshape(
            (num_query_groups // tp, (2 + num_querys_per_group*2)*dim, -1)
        )
        (
            q_proj_weight, 
            k_proj_weight, 
            v_proj_weight
        ) = torch.split(attn_proj_weight, [2*num_querys_per_group*dim, dim, dim], dim=1)

        q_proj_weight = q_proj_weight.reshape(num_query_groups // tp, 2, num_querys_per_group, dim, -1).transpose(1, 2).flatten(1, 3)
        self.copy(q_proj_weight, hf_attn.q_proj.weight, param_type=ParamType.QKV_W)
        self.copy(k_proj_weight, hf_attn.k_proj.weight, param_type=ParamType.QKV_W)
        self.copy(v_proj_weight, hf_attn.v_proj.weight, param_type=ParamType.QKV_W)

        self.copy(
            attn.linear_proj.weight,
            hf_attn.o_proj.weight,
            param_type=ParamType.ROW
        )

        # Copy bias
        if self.args.add_qkv_bias:
            attn_proj_bias = attn.linear_qkv.bias.reshape(
                (num_query_groups // tp, (2 + num_querys_per_group * 2)*dim, -1)
            )
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                attn_proj_bias, 
                [2*num_querys_per_group*dim, dim, dim], 
                dim=1
            )
            self.copy(q_proj_bias, hf_attn.q_proj.bias, param_type=ParamType.QKV_B)
            self.copy(k_proj_bias, hf_attn.k_proj.bias, param_type=ParamType.QKV_B)
            self.copy(v_proj_bias, hf_attn.v_proj.bias, param_type=ParamType.QKV_B)