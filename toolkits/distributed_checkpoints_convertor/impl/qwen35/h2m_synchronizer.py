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
import os
import torch
import logging
from typing import Dict
from general.h2m_synchronizer import HF2MGSynchronizer as _HF2MGSynchronizer
from general.synchronizer import ParamType

class HF2MGSynchronizer(_HF2MGSynchronizer):
    # TODO: to be refactored to Hybrid Model convertor
    def __init__(self, load_dir, model_provider_func=None):
        super().__init__(load_dir, model_provider_func)
        self.layout = self.get_hybrid_layout()
        

    def get_hybrid_layout(self) -> str:
        assert self.args.hybrid_override_pattern is not None
        return self.args.hybrid_override_pattern

    def set_aggregation_mlp_state(
        self, 
        mlp, 
        hf_mlp_gate_up_proj_weight, 
        hf_mlp_down_proj_weight,
        hf_mlp_gate_up_proj_bias,
        hf_mlp_down_proj_bias,
        expert_id='',
    ):
        '''
        Set MLP params.
        The mlp (mcore MLP) should have attributes `linear_fc1` and `linear_fc2`.
        Currently only Gated Linear is supported.
        '''
        if self.dryrun:
            gate_up_proj_weight = mlp.linear_fc1.weight
            if mlp.config.add_bias_linear:
                gate_up_proj_bias = mlp.linear_fc1.bias
        else:
            # 创建一个2的维度出来
            gate_up_proj_weight = hf_mlp_gate_up_proj_weight.reshape(2, hf_mlp_gate_up_proj_weight.shape[0] // 2, -1)
            if mlp.config.add_bias_linear:
                gate_up_proj_bias = hf_mlp_gate_up_proj_bias
        linear_fc1_weight = getattr(mlp.linear_fc1, f'weight{expert_id}')
        linear_fc2_weight = getattr(mlp.linear_fc2, f'weight{expert_id}')
        self.copy(
            gate_up_proj_weight, 
            linear_fc1_weight, 
            param_type=ParamType.GATE_UP if expert_id == '' else ParamType.MOE_GATE_UP
        )
        self.copy(
            hf_mlp_down_proj_weight, 
            linear_fc2_weight, 
            param_type=ParamType.ROW if expert_id == '' else ParamType.AGGREGATION_MOE_ROW
        )

        if mlp.config.add_bias_linear:
            linear_fc1_bias = getattr(mlp.linear_fc1, f'bias{expert_id}')
            linear_fc2_bias = getattr(mlp.linear_fc2, f'bias{expert_id}')
            self.copy(
                gate_up_proj_bias, 
                linear_fc1_bias, 
                param_type=ParamType.GATE_UP if expert_id == '' else ParamType.MOE_GATE_UP
            )
            self.copy(
                hf_mlp_down_proj_bias, 
                linear_fc2_bias,
            )

    def set_sequential_aggregation_mlp_state(self, experts, hf_experts):
        '''Set MOE MLP params.'''
        experts = experts.local_experts

        hf_experts_gate_up_proj = self.load_tensor(hf_experts.gate_up_proj)
        hf_experts_down_proj = self.load_tensor(hf_experts.down_proj)

        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            self.set_aggregation_mlp_state(
                experts[mg_expert_id], 
                hf_mlp_gate_up_proj_weight=hf_experts_gate_up_proj[hf_expert_id, :, :], 
                hf_mlp_down_proj_weight=hf_experts_down_proj[hf_expert_id, :, :], 
                hf_mlp_gate_up_proj_bias=None, 
                hf_mlp_down_proj_bias=None
            )

    def set_group_aggregation_mlp_state(self, experts, hf_experts):

        hf_experts_gate_up_proj = self.load_tensor(hf_experts.gate_up_proj)
        hf_experts_down_proj = self.load_tensor(hf_experts.down_proj)

        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            self.set_aggregation_mlp_state(
                experts, 
                hf_mlp_gate_up_proj_weight=hf_experts_gate_up_proj[hf_expert_id, :, :], 
                hf_mlp_down_proj_weight=hf_experts_down_proj[hf_expert_id, :, :], 
                hf_mlp_gate_up_proj_bias=None, 
                hf_mlp_down_proj_bias=None, 
                expert_id=mg_expert_id
            )

    def set_aggregation_moe_layer_state(self, moe, hf_moe):
        # router
        self.copy(hf_moe.gate.weight, moe.router.weight)
        if moe.router.enable_expert_bias:
            self.copy(hf_moe.gate.e_score_correction_bias, moe.router.expert_bias)
        # experts
        if self.args.moe_grouped_gemm:
            # group gemm
            if self.args.moe_use_legacy_grouped_gemm:
                # weight1 and weight2, not impl
                raise NotImplementedError("Currently only TE GroupGEMM is implemented.")
            self.set_group_aggregation_mlp_state(moe.experts, hf_moe.experts)
        else:
            # sequential
            self.set_sequential_aggregation_mlp_state(moe.experts, hf_moe.experts)

        # shared experts
        if moe.shared_experts is not None:
            if moe.shared_experts.use_shared_expert_gate:
                self.copy(hf_moe.shared_expert_gate.weight, moe.shared_experts.gate_weight)
            
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
            '''Set embedding params.'''
            self.copy(
                hf_model.model.language_model.embed_tokens.weight, 
                mg_model.embedding.word_embeddings.weight, 
                param_type=ParamType.COLUMN
            )

        if mg_model.post_process:
            self.copy(
                hf_model.model.language_model.norm.weight, 
                mg_model.decoder.final_norm.weight, 
            )
            
            if mg_model.share_embeddings_and_output_weights:
                output_layer_weight = mg_model.shared_embedding_or_output_weight() 
            else:
                output_layer_weight = mg_model.output_layer.weight
            self.copy(
                hf_model.lm_head.weight, 
                output_layer_weight, 
                param_type=ParamType.COLUMN
            )

            # mtp layer
            self.set_moe_layer_state(mg_model.mtp.layers[0].transformer_layer.mlp, hf_model.mtp.layers[0].mlp)
            self.copy(hf_model.mtp.layers[0].post_attention_layernorm.weight, mg_model.mtp.layers[0].transformer_layer.pre_mlp_layernorm.weight)
            self.set_gated_selfattn_state(mg_model.mtp.layers[0].transformer_layer.self_attention, hf_model.mtp.layers[0].self_attn)
            self.copy(hf_model.mtp.layers[0].input_layernorm.weight, mg_model.mtp.layers[0].transformer_layer.self_attention.linear_qgkv.layer_norm_weight)

            self.copy(hf_model.mtp.pre_fc_norm_embedding.weight, mg_model.mtp.layers[0].enorm.weight)
            self.copy(hf_model.mtp.pre_fc_norm_hidden.weight, mg_model.mtp.layers[0].hnorm.weight)
            self.copy(hf_model.mtp.norm.weight, mg_model.mtp.layers[0].final_layernorm.weight)

            self.copy(
                hf_model.mtp.fc.weight,
                mg_model.mtp.layers[0].eh_proj.weight,
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
                self.copy(hf_layer.input_layernorm.weight, layer.mixer.in_proj.layer_norm_weight)
            elif self.layout[global_mg_layer_id] == '-':
                # transformer_layer of MLP
                self.set_aggregation_moe_layer_state(layer.mlp, hf_layer.mlp)
                self.copy(hf_layer.post_attention_layernorm.weight, layer.pre_mlp_layernorm.weight)
            elif self.layout[global_mg_layer_id] == '*':
                # transformer_layer of Attention
                self.set_gated_selfattn_state(layer.self_attention, hf_layer.self_attn)
                self.copy(hf_layer.input_layernorm.weight, layer.self_attention.linear_qgkv.layer_norm_weight)
            else:
                raise ValueError(f"Unrecognized layer type {self.layout[global_mg_layer_id]} in {self.layout}")

    def set_mamba_layer_state(self, mixer, hf_mixer):
        # hf: qkvz+ba  mcore:zVKQba
        # Nk: num of key heads
        # Nv: num of value heads
        # Dk: dim of each key head
        # Dv: dim of each value head
        # linear_qkvz: [Nk * (2 * Dk + 2 * Dv * Nv // Nk), input_dim]
        # linear_ba: [Nk * (2 * Nv // Nk), input_dim]
        # in_proj: [
        #   (
        #       Nv // TP * Dv, # z
        #       Nv // TP * Dv, # V
        #       Nk // TP * Dk, # K
        #       Nk // TP * Dk, # Q
        #       Nv // TP, # b
        #       Nv // TP, # a
        #   ), 
        #   input_dim
        # ]
        Nk, Nv, Dk, Dv = (
            hf_mixer.num_k_heads,
            hf_mixer.num_v_heads,
            hf_mixer.head_k_dim,
            hf_mixer.head_v_dim
        )
        split_size_list = [
            Dk, 
            Dk, 
            Dv * Nv // Nk
        ]
        q, k, v = torch.split(
            self.load_tensor(hf_mixer.in_proj_qkv.weight).reshape(Nk, 2 * Dk + Dv * Nv // Nk, -1),
            split_size_list,
            dim=1
        )
        z = self.load_tensor(hf_mixer.in_proj_z.weight).reshape(Nk, Dv * Nv // Nk, -1)
        # print((
        #     z.shape, v.shape, q.shape, k.shape, 
        #     self.load_tensor(hf_mixer.in_proj_b.weight).reshape(Nk, Nv // Nk, -1).shape,
        #     self.load_tensor(hf_mixer.in_proj_b.weight).reshape(Nk, Nv // Nk, -1).shape,
        #     mixer.in_proj.weight.shape,
        # ))
        # TODO: GATE_UP is also a case of MERGED_LINEAR, with intermediate shape [hidden_size, 1, -1]
        self.copy([
            z,
            v,
            q,
            k,
            # *self.load_tensor(hf_mixer.in_proj_ba.weight).reshape(Nk, 2 * Nv // Nk, -1).chunk(chunks=2, dim=1)
            self.load_tensor(hf_mixer.in_proj_b.weight).reshape(Nk, Nv // Nk, -1),
            self.load_tensor(hf_mixer.in_proj_a.weight).reshape(Nk, Nv // Nk, -1),
        ], mixer.in_proj.weight, param_type=ParamType.MERGED_LINEAR)

        self.copy(hf_mixer.dt_bias, mixer.dt_bias, param_type=ParamType.COLUMN)
        self.copy(hf_mixer.A_log, mixer.A_log, param_type=ParamType.COLUMN)
        
        # hf: QKV [2 x Nk x Dk + Nv x Dv, 1, kernel_size]
        split_size_list = [
            Nk * Dk, 
            Nk * Dk, 
            Nv * Dv, 
        ]
        conv_q, conv_k, conv_v = torch.split(
            self.load_tensor(hf_mixer.conv1d.weight), 
            split_size_or_sections=split_size_list, 
            dim=0
        )
        self.copy([
            conv_v.reshape(Nk, Dv * Nv // Nk, 1, -1),
            conv_q.reshape(Nk, Dk, 1, -1),
            conv_k.reshape(Nk, Dk, 1, -1)
        ], mixer.conv1d.weight, param_type=ParamType.MERGED_LINEAR)

        self.copy(hf_mixer.norm.weight, mixer.norm.weight, param_type=ParamType.UNIQUE)
        self.copy(hf_mixer.out_proj.weight, mixer.out_proj.weight, param_type=ParamType.ROW)


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
        num_heads = self.args.num_attention_heads
        num_query_groups = (self.args.num_query_groups if self.args.group_query_attention else self.args.num_attention_heads)
        num_querys_per_group = num_heads // num_query_groups
        dim = self.args.kv_channels
        assert num_heads % num_querys_per_group == 0
        # copy qk norm if indeed.
        if self.args.qk_layernorm:
            self.copy(hf_attn.q_norm.weight, attn.q_layernorm.weight)
            self.copy(hf_attn.k_norm.weight, attn.k_layernorm.weight)

        # Copy weights (re-order dimensions for Megatron).
        if self.dryrun:
            attn_proj_weight = attn.linear_qgkv.weight
        else:
            kv_repeat_count = int(os.environ.get('KV_REPEAT_COUNT', '1'))

            tmp_k_proj_weight = self.load_tensor(hf_attn.k_proj.weight).reshape((num_query_groups // kv_repeat_count, 1, dim, -1))
            tmp_v_proj_weight = self.load_tensor(hf_attn.v_proj.weight).reshape((num_query_groups // kv_repeat_count, 1, dim, -1))

            print(tmp_k_proj_weight.shape, tmp_v_proj_weight.shape)

            tmp_k_proj_weight = torch.cat([tmp_k_proj_weight] * kv_repeat_count, dim=1)
            tmp_v_proj_weight = torch.cat([tmp_v_proj_weight] * kv_repeat_count, dim=1)

            print(tmp_k_proj_weight.shape, tmp_v_proj_weight.shape)

            attn_proj_weight = torch.cat([
                self.load_tensor(hf_attn.q_proj.weight).reshape((num_query_groups, num_querys_per_group, 2, dim, -1)).transpose(1, 2).flatten(1, 3),
                tmp_k_proj_weight.reshape((num_query_groups, dim, -1)),
                tmp_v_proj_weight.reshape((num_query_groups, dim, -1)),
            ], dim=1)
            
        self.copy(
            attn_proj_weight, 
            attn.linear_qgkv.weight,
            param_type=ParamType.QKV_W,
        )
        self.copy(
            hf_attn.o_proj.weight,
            attn.linear_proj.weight,
            param_type=ParamType.ROW
        )

        # Copy bias
        if self.args.add_qkv_bias:
            if self.dryrun:
                attn_proj_bias = attn.linear_qgkv.bias
            else:
                attn_proj_bias = torch.cat([
                    self.load_tensor(hf_attn.q_proj.bias).reshape((num_query_groups, num_querys_per_group*dim, -1)),
                    self.load_tensor(hf_attn.k_proj.bias).reshape((num_query_groups, dim, -1)),
                    self.load_tensor(hf_attn.v_proj.bias).reshape((num_query_groups, dim, -1)),
                ], dim=1)
            self.copy(
                attn_proj_bias, 
                attn.linear_qgkv.bias,
                param_type=ParamType.QKV_B,
            )