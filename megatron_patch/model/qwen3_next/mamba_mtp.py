
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.enums import AttnMaskType

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    make_viewless_tensor,
)

from megatron.core.transformer.multi_token_prediction import roll_tensor

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]

@dataclass
class MultiTokenPredictionLayerSubmodules:
    """
    Dataclass for specifying the submodules of a MultiTokenPrediction module.

    Args:
        hnorm (Union[ModuleSpec, type]): Specification or instance of the
             hidden states normalization to be applied.
        enorm (Union[ModuleSpec, type]): Specification or instance of the
            embedding normalization to be applied.
        eh_proj (Union[ModuleSpec, type]): Specification or instance of the
            linear projection to be applied.
        transformer_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer block to be applied.
    """

    enorm: Union[ModuleSpec, type] = None
    hnorm: Union[ModuleSpec, type] = None
    eh_proj: Union[ModuleSpec, type] = None
    transformer_layer: Union[ModuleSpec, type] = None
    layer_norm: Union[ModuleSpec, type] = None



class MultiTokenPredictionLayer(MegatronModule):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiTokenPredictionLayerSubmodules,
        layer_number: int = 1,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.sequence_parallel = config.sequence_parallel
        self.submodules = submodules
        self.layer_number = layer_number
        assert pg_collection is not None, "pg_collection must be provided for MambaMTP"
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp

        self_attention_spec = self.submodules.transformer_layer.submodules.self_attention
        attn_mask_type = self_attention_spec.params.get('attn_mask_type', '')
        assert attn_mask_type in SUPPORTED_ATTN_MASK, (
            f"Multi-Token Prediction (MTP) is not jet supported with "
            + f"{attn_mask_type} attention mask type."
            + f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
        )

        self.enorm = build_module(
            self.submodules.enorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
        # of the i-th tocken's hidden states and the (i + K)-th tocken's decoder input,
        # so the input's shape is [s, b, 2*h].
        # The output will be send to the following transformer layer,
        # so the output's shape should be [s, b, h].
        self.eh_proj = build_module(
            self.submodules.eh_proj,
            self.config.hidden_size * 2,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )
        self.transformer_layer = build_module(
            self.submodules.transformer_layer, 
            config=self.config, 
            pg_collection=self.pg_collection,
        )

        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.offload_context = nullcontext()

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
    ):
        """
        Preprocesses input data for the Multi-Token Prediction (MTP) layers.

        This function computes the decoder input and sends updated input_ids and position_ids to
        the next layer.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            position_ids (torch.Tensor): The position IDs corresponding to the input tokens.
            embedding (Callable): The embedding module
                from gpt model to compute the decoder input.
            hidden_states (torch.Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
        """
        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1, cp_group=self.cp_group)
        position_ids, _ = roll_tensor(position_ids, shifts=-1, dims=-1, cp_group=self.cp_group)
        # embedding
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return input_ids, position_ids, decoder_input, hidden_states

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """
        Concatenate the tokens before sending to transformer layer.
        """
        decoder_input = self.enorm(decoder_input)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
        # and the (i + K)-th tocken's embedding, and combine them with linear projection.
        hidden_states = torch.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel we need to gather the tensor across the model-parallel
        # ranks after the linear projection. This used to call
        # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
        # the gradient in backward pass and was therefore incorrect in this context.
        # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
        hidden_states = gather_from_tensor_model_parallel_region(hidden_states)
        # For sequence parallel, scatter after linear_fc and before transformer layer.
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        return hidden_states

    def _proj_and_transformer_layer(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concatenates embeddings with hidden states and then applies transformer layer forward.
        """
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Unlike transformer_block.py which needs to support mixed-precision in
        # different layers,currently MTP only use global fp8 context.
        if self.config.fp8:
            fp8_context = get_fp8_context(self.config)
            transformer_layer_fp8_context = get_fp8_context(self.config)
        else:
            fp8_context = nullcontext()
            transformer_layer_fp8_context = nullcontext()

        with rng_context:
            with fp8_context:
                hidden_states = self._concat_embeddings(hidden_states, decoder_input)

            # Use a separate fp8 context for the transformer layer. This is to ensure that when the
            # transformer layer is cudagraphed, the FP8GlobalStateManager.is_first_fp8_module() is
            # True so that the fp8 weight caching can be triggered correctly.
            # print("MTP: packed_seq_params", packed_seq_params)
            with transformer_layer_fp8_context:
                hidden_states, _ = self.transformer_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )

        hidden_states = self._postprocess(hidden_states)

        return hidden_states

    def _postprocess(self, hidden_states: torch.Tensor):
        """
        Postprocesses the output of the transformer layers.
        """

        # Layer norm before shared head layer.
        hidden_states = self.final_layernorm(hidden_states)
        # TENorm produces a "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return hidden_states

    def _checkpointed_forward(
        self,
        forward_func,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
    ):
        def custom_forward(
            hidden_states, attention_mask, context, context_mask, rotary_pos_emb
        ):
            return forward_func(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        def checkpoint_handler():
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                return te_checkpoint(
                    custom_forward,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    custom_forward, 
                    self.config.distribute_saved_activations, 
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert (
                self.config.recompute_num_layers == 1
            ), "recompute_num_layers must be 1 for MTP recompute"
            outputs = checkpoint_handler()
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
        embedding=None,
    ):
        """
        Execute the forward pass through the Multi-Token Prediction (MTP) layer.

        Args:
            input_ids (Tensor): Input token IDs .
            position_ids (Tensor): Positional IDs of the input tokens.
            hidden_states (Tensor): Hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention, if applicable.
            context_mask (Tensor, optional): Mask for cross-attention context, if applicable.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Cosine component of rotary positional embeddings.
            rotary_pos_sin (Tensor, optional): Sine component of rotary positional embeddings.
            sequence_len_offset (Tensor, optional): Offset for sequence length, if applicable.
            embedding (Callable): The embedding module from gpt model to compute the decoder input.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        assert context is None, f"multi token prediction + cross attention is not yet supported."
        # assert (
        #     packed_seq_params is None
        # ), f"multi token prediction + sequence packing is not yet supported."

        input_ids, position_ids, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
        )

        # # recompute packed_seq_params
        # if packed_seq_params is not None:
        #     with torch.no_grad():
        #         position_ids = position_ids[0] # shape: [seq_length]
        #         # add a zero column to the position_ids
        #         start_indices = (torch.cat([torch.zeros_like(position_ids[:1]), position_ids[1:]], dim=0) == 0).nonzero(as_tuple=True)[0]
        #         seqlens = start_indices[1:] - start_indices[:-1]
        #         # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
        #         cu_seqlens = torch.zeros(start_indices.shape[0] + 1, device=position_ids.device, dtype=torch.int)
        #         cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
        #         cu_seqlens[-1] = position_ids.shape[0]
        #         # fix position_ids = [0,1,...,max] case
        #         if seqlens.shape[0] == 0:
        #             max_seqlen = position_ids.max() + 1
        #         else:
        #             max_seqlen = torch.max(seqlens.max(), position_ids.max() + 1)
        #         packed_seq_params = PackedSeqParams(
        #             cu_seqlens_q=cu_seqlens,
        #             cu_seqlens_kv=cu_seqlens,
        #             qkv_format='thd',
        #             max_seqlen_q = max_seqlen,
        #             max_seqlen_kv = max_seqlen,
        #         )

        # print("MTP: position_ids", position_ids)
        
        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                self._proj_and_transformer_layer,
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        return hidden_states, input_ids, position_ids

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        return sharded_state_dict