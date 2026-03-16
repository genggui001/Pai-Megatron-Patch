# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import os

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name

from qwen3_base_energon_helpers import MyTaskEncoder, print_error_handler


def build_to_nex_input(max_seq_length, input_pad_token_id, pos_pad_token_id, label_pad_token_id):
    def to_nex_input(features):
        sequence_length = features["input_ids"].shape[-1]
        
        return {
            "tokens": F.pad(features["input_ids"], (0, max_seq_length - sequence_length), "constant", input_pad_token_id),
            # "position_ids": F.pad(features["position_ids"], (0, max_seq_length - sequence_length), "constant", pos_pad_token_id),
            "position_ids": torch.cat([features["position_ids"], torch.arange(0, max_seq_length-sequence_length, dtype=torch.long)[None, :]], dim=-1),
            "labels": F.pad(features["labels"], (0, max_seq_length - sequence_length), "constant", label_pad_token_id),
            "loss_mask": F.pad(features["loss_mask"], (0, max_seq_length - sequence_length), "constant", 0.0),
        }

    return to_nex_input


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""
    def __init__(self, dataloader, deal_fn):
        self._dataloader = dataloader
        self._deal_fn = deal_fn
        self._iter = iter(cyclic_iter(self._dataloader, self._deal_fn))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


def cyclic_iter(iter, deal_fn):
    while True:
        for x in iter:
            yield deal_fn(x)
        
        print("----------- one dataloader restart -----------")



# def is_first_or_last_stage(pp_size):
#     """Check if the current pipeline parallel stage is the first or last stage."""
#     if pp_size == 1:    # No pipeline parallelism.
#         return True

#     # With no separate pipeline stage for the vision model (epp=0), 
#     # run the dataloader on the first and last pipeline stage.
#     pp_rank = get_pipeline_model_parallel_rank()
#     is_valid_rank = pp_rank in (0, pp_size-1)

#     return is_valid_rank


def is_dataloader_rank():
    """Check if we should have the dataloader on this tensor and pipeline parallel rank."""
    # Run dataloader only on the first tensor parallel rank (will be broadcasted to others).
    is_first_rank = get_tensor_model_parallel_rank() == 0

    # pp_size = get_pipeline_model_parallel_world_size()
    # is_first_rank = is_first_rank and is_first_or_last_stage(pp_size)

    return is_first_rank


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()

    assert args.micro_batch_size == 1
    assert args.patch_tokenizer_path is not None

    # Dataloader is only on specific ranks.
    if not is_dataloader_rank():
        return None, None, None

    worker_debug_path = None
    worker_log_level = 0
    dname = args.data_path[0] if type(args.data_path) is list else args.data_path

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()

    print(("dataloader seed", args.seed))

    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        seed_offset=args.seed,
        data_parallel_group=data_parallel_group,
        worker_debug_path=worker_debug_path,
        worker_log_level=worker_log_level,
    )

    train_ds = get_train_dataset(
        dname,
        split_part="train",
        task_encoder=MyTaskEncoder(
            tokenizer_path=args.patch_tokenizer_path,
            seq_length=args.seq_length,
            sensitive_words_path=None,
        ),
        batch_size=1,
        packing_buffer_size=8192,
        shuffle_buffer_size=8192,
        max_samples_per_sequence=None,
        worker_config=worker_config,
        handler=print_error_handler,
    )
    val_ds = get_train_dataset(
        dname,
        split_part="val",
        task_encoder=MyTaskEncoder(
            tokenizer_path=args.patch_tokenizer_path,
            seq_length=args.seq_length,
            sensitive_words_path=None,
        ),
        batch_size=1,
        packing_buffer_size=8192,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        worker_config=worker_config,
        handler=print_error_handler,
    )
    val_ds = LimitDataset(
        val_ds,
        length=args.eval_iters * get_num_microbatches(),
        worker_config=worker_config,
        reset_after_epoch=True,
    )

    train_dataloader = get_savable_loader(train_ds, worker_config=worker_config, watchdog_timeout_seconds=240)
    valid_dataloader = get_loader(val_ds, worker_config=worker_config, watchdog_timeout_seconds=240)

    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                pipeline_rank=0,    # Only the first pipeline parallel rank stores the dataloader checkpoint.
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(data_save_name, map_location="cpu")
                    train_dataloader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
                    print(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print("loading dataset state failed. Skipping. " + str(e))
            else:
                print(f"dataset state {data_save_name} does not exist")

    input_pad_token_id: int = 151643 # <|endoftext|>
    pos_pad_token_id: int = 0
    label_pad_token_id: int = -100
        
    deal_fn = build_to_nex_input(args.seq_length, input_pad_token_id, pos_pad_token_id, label_pad_token_id)

    return EnergonDataloader(train_dataloader, deal_fn), [EnergonDataloader(valid_dataloader, deal_fn)], None

