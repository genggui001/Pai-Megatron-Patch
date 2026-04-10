from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only when Triton is unavailable.
    triton = None
    tl = None
    HAS_TRITON = False


__all__ = [
    "HAS_TRITON",
    "all_reduce_autograd",
    "torch_reference_grouped_kl",
    "torch_reference_grouped_kl_tp",
    "triton_grouped_kl",
    "triton_grouped_kl_tp",
]


class _AllReduce(torch.autograd.Function):
    """All-reduce that preserves backward propagation."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group, ctx.op = group, op
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output))


def all_reduce_autograd(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return tensor
    return _AllReduce.apply(op, group, tensor)


def _tp_world_size(group: Optional[dist.ProcessGroup]) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def _flatten_grouped_shape(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    leading_shape = tensor.shape[:-1]
    return tensor.reshape(-1, tensor.shape[-1]).contiguous(), leading_shape


def _compute_rowwise_tp_stats(
    student_2d: torch.Tensor,
    teacher_2d: torch.Tensor,
    *,
    temperature: float,
    tp_group: Optional[dist.ProcessGroup],
    autograd_student_denom: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if HAS_TRITON and student_2d.is_cuda and teacher_2d.is_cuda and not autograd_student_denom:
        teacher_local_max, teacher_exp_sum = _rowwise_online_logsumexp_triton(
            teacher_2d, temperature=temperature
        )
        student_local_max, student_exp_sum = _rowwise_online_logsumexp_triton(
            student_2d, temperature=temperature
        )

        if _tp_world_size(tp_group) > 1:
            teacher_row_max = teacher_local_max.clone()
            student_row_max = student_local_max.clone()
            dist.all_reduce(teacher_row_max, op=dist.ReduceOp.MAX, group=tp_group)
            dist.all_reduce(student_row_max, op=dist.ReduceOp.MAX, group=tp_group)
            teacher_exp_sum.mul_(torch.exp(teacher_local_max - teacher_row_max))
            student_exp_sum.mul_(torch.exp(student_local_max - student_row_max))
            dist.all_reduce(teacher_exp_sum, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(student_exp_sum, op=dist.ReduceOp.SUM, group=tp_group)
        else:
            teacher_row_max = teacher_local_max
            student_row_max = student_local_max

        student_log_denom = torch.log(student_exp_sum)
        teacher_log_denom = torch.log(teacher_exp_sum)
        return student_row_max, teacher_row_max, student_log_denom, teacher_log_denom

    output_teacher = teacher_2d.detach().float().clone()
    output_teacher.div_(temperature)
    output_student = student_2d.float().clone()
    output_student.div_(temperature)

    teacher_row_max, _ = torch.max(output_teacher, dim=-1)
    student_row_max, _ = torch.max(output_student, dim=-1)

    if _tp_world_size(tp_group) > 1:
        dist.all_reduce(teacher_row_max, op=dist.ReduceOp.MAX, group=tp_group)
        dist.all_reduce(student_row_max, op=dist.ReduceOp.MAX, group=tp_group)

    output_teacher.sub_(teacher_row_max.unsqueeze(dim=-1))
    output_student.sub_(student_row_max.unsqueeze(dim=-1).detach())

    denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
    denom_student = torch.sum(torch.exp(output_student), dim=-1)

    if _tp_world_size(tp_group) > 1:
        dist.all_reduce(denom_teacher, op=dist.ReduceOp.SUM, group=tp_group)
        if autograd_student_denom:
            denom_student = all_reduce_autograd(denom_student, group=tp_group)
        else:
            dist.all_reduce(denom_student, op=dist.ReduceOp.SUM, group=tp_group)

    student_log_denom = torch.log(denom_student)
    teacher_log_denom = torch.log(denom_teacher)
    return student_row_max, teacher_row_max, student_log_denom, teacher_log_denom


def _compute_student_tp_stats_only(
    student_2d: torch.Tensor,
    *,
    temperature: float,
    tp_group: Optional[dist.ProcessGroup],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if HAS_TRITON and student_2d.is_cuda:
        student_local_max, student_exp_sum = _rowwise_online_logsumexp_triton(
            student_2d, temperature=temperature
        )
        if _tp_world_size(tp_group) > 1:
            student_row_max = student_local_max.clone()
            dist.all_reduce(student_row_max, op=dist.ReduceOp.MAX, group=tp_group)
            student_exp_sum.mul_(torch.exp(student_local_max - student_row_max))
            dist.all_reduce(student_exp_sum, op=dist.ReduceOp.SUM, group=tp_group)
        else:
            student_row_max = student_local_max
        student_log_denom = torch.log(student_exp_sum)
        return student_row_max, student_log_denom

    output_student = student_2d.float().clone()
    output_student.div_(temperature)

    student_row_max, _ = torch.max(output_student, dim=-1)
    if _tp_world_size(tp_group) > 1:
        dist.all_reduce(student_row_max, op=dist.ReduceOp.MAX, group=tp_group)

    output_student.sub_(student_row_max.unsqueeze(dim=-1).detach())
    denom_student = torch.sum(torch.exp(output_student), dim=-1)
    if _tp_world_size(tp_group) > 1:
        dist.all_reduce(denom_student, op=dist.ReduceOp.SUM, group=tp_group)

    student_log_denom = torch.log(denom_student)
    return student_row_max, student_log_denom


def _validate_inputs(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher shape mismatch: {student_logits.shape} vs {teacher_logits.shape}"
        )
    if student_logits.ndim < 2:
        raise ValueError("expected logits with shape [..., vocab], got rank < 2")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    leading_shape = student_logits.shape[:-1]
    student_2d = student_logits.reshape(-1, student_logits.shape[-1]).contiguous()
    teacher_2d = teacher_logits.reshape(-1, teacher_logits.shape[-1]).contiguous()
    return student_2d, teacher_2d, leading_shape


def torch_reference_grouped_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    reverse: bool = False,
) -> torch.Tensor:
    """Reference grouped KL using PyTorch's native `F.kl_div`.

    This matches the semantics used in `TensorParallelLogitsKLLoss` on a single
    local shard: student logits are Q, teacher logits are P, and
    `reverse=False` means forward KL `KL(P || Q)`.
    """

    student_2d, teacher_2d, leading_shape = _validate_inputs(
        student_logits, teacher_logits, temperature
    )

    output_teacher = teacher_2d.detach().float().clone()
    output_teacher.div_(temperature)
    output_student = student_2d.float().clone()
    output_student.div_(temperature)

    teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
    output_teacher.sub_(teacher_logits_max.unsqueeze(dim=-1))
    denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)

    student_logits_max, _ = torch.max(output_student, dim=-1)
    output_student.sub_(student_logits_max.unsqueeze(dim=-1).detach())
    denom_student = torch.sum(torch.exp(output_student), dim=-1)

    denom_teacher.log_()
    output_teacher.sub_(denom_teacher.unsqueeze(dim=-1))

    denom_student.log_()
    output_student.sub_(denom_student.unsqueeze(dim=-1))

    if reverse:
        loss = torch.sum(
            F.kl_div(
                output_teacher,
                output_student,
                reduction="none",
                log_target=True,
            ),
            dim=-1,
        )
    else:
        loss = torch.sum(
            F.kl_div(
                output_student,
                output_teacher,
                reduction="none",
                log_target=True,
            ),
            dim=-1,
        )

    return loss.reshape(leading_shape)


def torch_reference_grouped_kl_tp(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    reverse: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """TP-aware PyTorch reference matching current KL distributed semantics.

    The returned tensor is still the local shard contribution with shape
    `student_logits.shape[:-1]`. Correct student gradients are preserved via
    `all_reduce_autograd` on the student denominator.
    """

    student_2d, teacher_2d, leading_shape = _validate_inputs(
        student_logits, teacher_logits, temperature
    )
    (
        student_row_max,
        teacher_row_max,
        student_log_denom,
        teacher_log_denom,
    ) = _compute_rowwise_tp_stats(
        student_2d,
        teacher_2d,
        temperature=temperature,
        tp_group=tp_group,
        autograd_student_denom=True,
    )

    output_teacher = teacher_2d.detach().float().clone()
    output_teacher.div_(temperature)
    output_teacher.sub_(teacher_row_max.unsqueeze(dim=-1))
    output_teacher.sub_(teacher_log_denom.unsqueeze(dim=-1))

    output_student = student_2d.float().clone()
    output_student.div_(temperature)
    output_student.sub_(student_row_max.unsqueeze(dim=-1).detach())
    output_student.sub_(student_log_denom.unsqueeze(dim=-1))

    if reverse:
        loss = torch.sum(
            F.kl_div(
                output_teacher,
                output_student,
                reduction="none",
                log_target=True,
            ),
            dim=-1,
        )
    else:
        loss = torch.sum(
            F.kl_div(
                output_student,
                output_teacher,
                reduction="none",
                log_target=True,
            ),
            dim=-1,
        )

    return loss.reshape(leading_shape)


if HAS_TRITON:
    _KERNEL_CONFIGS = [
        triton.Config({"BLOCK_SIZE_V": block_size}, num_warps=num_warps, num_stages=stages)
        for block_size, num_warps, stages in (
            (256, 2, 2),
            (256, 4, 2),
            (512, 4, 2),
            (512, 8, 2),
            (1024, 8, 2),
            (1024, 8, 4),
            (2048, 8, 2),
            (2048, 8, 4),
        )
    ]

    _2D_KERNEL_CONFIGS = [
        triton.Config({"BLOCK_SIZE_V": block_size}, num_warps=num_warps, num_stages=stages)
        for block_size, num_warps, stages in (
            (256, 4, 2),
            (512, 4, 2),
            (512, 8, 2),
            (1024, 8, 2),
            (1024, 8, 4),
        )
    ]

    @triton.autotune(configs=_KERNEL_CONFIGS, key=["num_cols"])
    @triton.jit
    def _rowwise_online_logsumexp_kernel(
        input_ptr,
        out_max_ptr,
        out_exp_sum_ptr,
        num_rows,
        num_cols,
        stride_input_row,
        inv_temperature,
        BLOCK_SIZE_V: tl.constexpr,
    ):
        """Single-pass online softmax: computes row max and exp-sum simultaneously."""
        row_idx = tl.program_id(0).to(tl.int64)
        if row_idx >= num_rows:
            return

        row_ptr = input_ptr + row_idx * stride_input_row
        neg_inf = float("-inf")
        row_max = neg_inf
        row_exp_sum = 0.0

        for start in tl.range(0, num_cols, BLOCK_SIZE_V):
            cols = (start + tl.arange(0, BLOCK_SIZE_V)).to(tl.int64)
            mask = cols < num_cols
            vals = tl.load(row_ptr + cols, mask=mask, other=neg_inf).to(tl.float32)
            vals = vals * inv_temperature
            block_max = tl.max(vals, axis=0)
            new_max = tl.maximum(row_max, block_max)
            row_exp_sum = (
                row_exp_sum * tl.exp(row_max - new_max)
                + tl.sum(tl.exp(vals - new_max), axis=0)
            )
            row_max = new_max

        tl.store(out_max_ptr + row_idx, row_max)
        tl.store(out_exp_sum_ptr + row_idx, row_exp_sum)

    @triton.autotune(configs=_KERNEL_CONFIGS, key=["num_cols", "reverse"])
    @triton.jit
    def _grouped_kl_kernel(
        student_ptr,
        teacher_ptr,
        output_ptr,
        num_rows,
        num_cols,
        stride_student_row,
        stride_teacher_row,
        stride_output_row,
        inv_temperature,
        reverse: tl.constexpr,
        BLOCK_SIZE_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        if row_idx >= num_rows:
            return

        student_row_ptr = student_ptr + row_idx * stride_student_row
        teacher_row_ptr = teacher_ptr + row_idx * stride_teacher_row

        neg_inf = float("-inf")

        # Pass 1: fused online softmax — compute max and exp-sum in a single pass
        student_max = neg_inf
        teacher_max = neg_inf
        student_denom = 0.0
        teacher_denom = 0.0

        for start in tl.range(0, num_cols, BLOCK_SIZE_V):
            cols = (start + tl.arange(0, BLOCK_SIZE_V)).to(tl.int64)
            mask = cols < num_cols
            student_vals = tl.load(student_row_ptr + cols, mask=mask, other=neg_inf).to(tl.float32)
            teacher_vals = tl.load(teacher_row_ptr + cols, mask=mask, other=neg_inf).to(tl.float32)
            student_vals = student_vals * inv_temperature
            teacher_vals = teacher_vals * inv_temperature

            new_s_max = tl.maximum(student_max, tl.max(student_vals, axis=0))
            student_denom = (
                student_denom * tl.exp(student_max - new_s_max)
                + tl.sum(tl.exp(student_vals - new_s_max), axis=0)
            )
            student_max = new_s_max

            new_t_max = tl.maximum(teacher_max, tl.max(teacher_vals, axis=0))
            teacher_denom = (
                teacher_denom * tl.exp(teacher_max - new_t_max)
                + tl.sum(tl.exp(teacher_vals - new_t_max), axis=0)
            )
            teacher_max = new_t_max

        student_log_denom = tl.log(student_denom)
        teacher_log_denom = tl.log(teacher_denom)

        # Pass 2: compute KL divergence
        kl_acc = 0.0
        for start in tl.range(0, num_cols, BLOCK_SIZE_V):
            cols = (start + tl.arange(0, BLOCK_SIZE_V)).to(tl.int64)
            mask = cols < num_cols
            student_vals = tl.load(student_row_ptr + cols, mask=mask, other=neg_inf).to(tl.float32)
            teacher_vals = tl.load(teacher_row_ptr + cols, mask=mask, other=neg_inf).to(tl.float32)
            student_vals = student_vals * inv_temperature
            teacher_vals = teacher_vals * inv_temperature

            student_log_prob = student_vals - student_max - student_log_denom
            teacher_log_prob = teacher_vals - teacher_max - teacher_log_denom

            if reverse:
                row_kl = tl.exp(student_log_prob) * (student_log_prob - teacher_log_prob)
            else:
                row_kl = tl.exp(teacher_log_prob) * (teacher_log_prob - student_log_prob)
            kl_acc += tl.sum(tl.where(mask, row_kl, 0.0), axis=0)

        tl.store(output_ptr + row_idx * stride_output_row, kl_acc)

    @triton.autotune(configs=_KERNEL_CONFIGS, key=["num_cols", "reverse"])
    @triton.jit
    def _grouped_kl_from_stats_kernel(
        student_ptr,
        teacher_ptr,
        student_row_max_ptr,
        teacher_row_max_ptr,
        student_log_denom_ptr,
        teacher_log_denom_ptr,
        output_ptr,
        num_rows,
        num_cols,
        stride_student_row,
        stride_teacher_row,
        stride_output_row,
        inv_temperature,
        reverse: tl.constexpr,
        BLOCK_SIZE_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        if row_idx >= num_rows:
            return

        student_row_ptr = student_ptr + row_idx * stride_student_row
        teacher_row_ptr = teacher_ptr + row_idx * stride_teacher_row
        student_row_max = tl.load(student_row_max_ptr + row_idx)
        teacher_row_max = tl.load(teacher_row_max_ptr + row_idx)
        student_log_denom = tl.load(student_log_denom_ptr + row_idx)
        teacher_log_denom = tl.load(teacher_log_denom_ptr + row_idx)

        kl_acc = 0.0
        for start in tl.range(0, num_cols, BLOCK_SIZE_V):
            cols = (start + tl.arange(0, BLOCK_SIZE_V)).to(tl.int64)
            mask = cols < num_cols

            student_vals = tl.load(student_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            teacher_vals = tl.load(teacher_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            student_log_prob = student_vals * inv_temperature - student_row_max - student_log_denom
            teacher_log_prob = teacher_vals * inv_temperature - teacher_row_max - teacher_log_denom

            if reverse:
                row_kl = tl.exp(student_log_prob) * (student_log_prob - teacher_log_prob)
            else:
                row_kl = tl.exp(teacher_log_prob) * (teacher_log_prob - student_log_prob)
            kl_acc += tl.sum(tl.where(mask, row_kl, 0.0), axis=0)

        tl.store(output_ptr + row_idx * stride_output_row, kl_acc)

    @triton.autotune(configs=_2D_KERNEL_CONFIGS, key=["num_cols"])
    @triton.jit
    def _grouped_kl_backward_kernel(
        student_ptr,
        teacher_ptr,
        student_row_max_ptr,
        student_log_denom_ptr,
        global_kl_ptr,
        grad_output_ptr,
        grad_student_ptr,
        num_rows,
        num_cols,
        stride_student_row,
        stride_teacher_row,
        stride_grad_student_row,
        inv_temperature,
        BLOCK_SIZE_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        if row_idx >= num_rows:
            return

        start = block_idx * BLOCK_SIZE_V
        cols = (start + tl.arange(0, BLOCK_SIZE_V)).to(tl.int64)
        mask = cols < num_cols

        student_row_ptr = student_ptr + row_idx * stride_student_row
        teacher_row_ptr = teacher_ptr + row_idx * stride_teacher_row
        grad_row_ptr = grad_student_ptr + row_idx * stride_grad_student_row

        student_row_max = tl.load(student_row_max_ptr + row_idx)
        student_log_denom = tl.load(student_log_denom_ptr + row_idx)
        grad_output = tl.load(grad_output_ptr + row_idx)
        global_kl = tl.load(global_kl_ptr + row_idx)

        student_vals = tl.load(student_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        teacher_log_prob = tl.load(teacher_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        student_log_prob = student_vals * inv_temperature - student_row_max - student_log_denom
        student_prob = tl.exp(student_log_prob)

        grad_vals = student_prob * (student_log_prob - teacher_log_prob - global_kl)
        grad_vals = grad_vals * grad_output * inv_temperature
        tl.store(grad_row_ptr + cols, grad_vals, mask=mask)

    @triton.autotune(configs=_2D_KERNEL_CONFIGS, key=["num_cols"])
    @triton.jit
    def _grouped_kl_backward_forward_kernel(
        student_ptr,
        teacher_prob_ptr,
        student_row_max_ptr,
        student_log_denom_ptr,
        grad_output_ptr,
        grad_student_ptr,
        num_rows,
        num_cols,
        stride_student_row,
        stride_teacher_prob_row,
        stride_grad_student_row,
        inv_temperature,
        BLOCK_SIZE_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        if row_idx >= num_rows:
            return

        start = block_idx * BLOCK_SIZE_V
        cols = (start + tl.arange(0, BLOCK_SIZE_V)).to(tl.int64)
        mask = cols < num_cols

        student_row_ptr = student_ptr + row_idx * stride_student_row
        teacher_prob_row_ptr = teacher_prob_ptr + row_idx * stride_teacher_prob_row
        grad_row_ptr = grad_student_ptr + row_idx * stride_grad_student_row

        student_row_max = tl.load(student_row_max_ptr + row_idx)
        student_log_denom = tl.load(student_log_denom_ptr + row_idx)
        grad_output = tl.load(grad_output_ptr + row_idx)

        student_vals = tl.load(student_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        teacher_prob = tl.load(teacher_prob_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        student_log_prob = student_vals * inv_temperature - student_row_max - student_log_denom
        student_prob = tl.exp(student_log_prob)
        grad_vals = (student_prob - teacher_prob) * grad_output * inv_temperature
        tl.store(grad_row_ptr + cols, grad_vals, mask=mask)

    @triton.autotune(configs=_KERNEL_CONFIGS, key=["num_cols"])
    @triton.jit
    def _grouped_reverse_kl_local_kernel(
        student_ptr,
        teacher_log_prob_ptr,
        student_row_max_ptr,
        student_log_denom_ptr,
        output_ptr,
        num_rows,
        num_cols,
        stride_student_row,
        stride_teacher_log_prob_row,
        stride_output_row,
        inv_temperature,
        BLOCK_SIZE_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        if row_idx >= num_rows:
            return

        student_row_ptr = student_ptr + row_idx * stride_student_row
        teacher_log_prob_row_ptr = teacher_log_prob_ptr + row_idx * stride_teacher_log_prob_row
        student_row_max = tl.load(student_row_max_ptr + row_idx)
        student_log_denom = tl.load(student_log_denom_ptr + row_idx)

        kl_acc = 0.0
        for start in tl.range(0, num_cols, BLOCK_SIZE_V):
            cols = (start + tl.arange(0, BLOCK_SIZE_V)).to(tl.int64)
            mask = cols < num_cols

            student_vals = tl.load(student_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            teacher_log_prob = tl.load(
                teacher_log_prob_row_ptr + cols, mask=mask, other=0.0
            ).to(tl.float32)
            student_log_prob = student_vals * inv_temperature - student_row_max - student_log_denom
            student_prob = tl.exp(student_log_prob)
            row_kl = student_prob * (student_log_prob - teacher_log_prob)
            kl_acc += tl.sum(tl.where(mask, row_kl, 0.0), axis=0)

        tl.store(output_ptr + row_idx * stride_output_row, kl_acc)

    @triton.autotune(configs=_2D_KERNEL_CONFIGS, key=["num_cols", "store_exp"])
    @triton.jit
    def _teacher_cache_from_stats_kernel(
        teacher_ptr,
        teacher_row_max_ptr,
        teacher_log_denom_ptr,
        output_ptr,
        num_rows,
        num_cols,
        stride_teacher_row,
        stride_output_row,
        inv_temperature,
        store_exp: tl.constexpr,
        BLOCK_SIZE_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        if row_idx >= num_rows:
            return

        start = block_idx * BLOCK_SIZE_V
        cols = (start + tl.arange(0, BLOCK_SIZE_V)).to(tl.int64)
        mask = cols < num_cols

        teacher_row_ptr = teacher_ptr + row_idx * stride_teacher_row
        output_row_ptr = output_ptr + row_idx * stride_output_row
        teacher_row_max = tl.load(teacher_row_max_ptr + row_idx)
        teacher_log_denom = tl.load(teacher_log_denom_ptr + row_idx)

        teacher_vals = tl.load(teacher_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        teacher_log_prob = teacher_vals * inv_temperature - teacher_row_max - teacher_log_denom
        output_vals = tl.exp(teacher_log_prob) if store_exp else teacher_log_prob
        tl.store(output_row_ptr + cols, output_vals, mask=mask)


def _rowwise_online_logsumexp_triton(
    logits_2d: torch.Tensor, *, temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (row_max, exp_sum) computed via single-pass online softmax."""
    num_rows = logits_2d.shape[0]
    row_max = torch.empty(num_rows, device=logits_2d.device, dtype=torch.float32)
    exp_sum = torch.empty(num_rows, device=logits_2d.device, dtype=torch.float32)
    _rowwise_online_logsumexp_kernel[(num_rows,)](
        logits_2d,
        row_max,
        exp_sum,
        num_rows,
        logits_2d.shape[1],
        logits_2d.stride(0),
        1.0 / temperature,
    )
    return row_max, exp_sum


def _build_teacher_cache_triton(
    teacher_2d: torch.Tensor,
    teacher_row_max: torch.Tensor,
    teacher_log_denom: torch.Tensor,
    *,
    temperature: float,
    store_exp: bool,
) -> torch.Tensor:
    output = torch.empty_like(teacher_2d)
    num_rows, num_cols = teacher_2d.shape
    grid = lambda meta: (num_rows, triton.cdiv(num_cols, meta["BLOCK_SIZE_V"]))
    _teacher_cache_from_stats_kernel[grid](
        teacher_2d,
        teacher_row_max,
        teacher_log_denom,
        output,
        num_rows,
        num_cols,
        teacher_2d.stride(0),
        output.stride(0),
        1.0 / temperature,
        store_exp=store_exp,
    )
    return output


class TritonGroupedKLFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0,
        reverse: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        if not HAS_TRITON:
            raise RuntimeError("Triton is not available in the current environment.")

        student_2d, teacher_2d, leading_shape = _validate_inputs(
            student_logits, teacher_logits, temperature
        )
        if student_2d.device.type != "cuda" or teacher_2d.device.type != "cuda":
            raise ValueError("triton_grouped_kl_tp expects CUDA tensors.")
        if student_2d.device != teacher_2d.device:
            raise ValueError("student and teacher must be on the same CUDA device.")

        (
            student_row_max,
            teacher_row_max,
            student_log_denom,
            teacher_log_denom,
        ) = _compute_rowwise_tp_stats(
            student_2d,
            teacher_2d,
            temperature=temperature,
            tp_group=tp_group,
            autograd_student_denom=False,
        )

        local_loss = torch.empty(student_2d.shape[0], device=student_2d.device, dtype=torch.float32)
        _grouped_kl_from_stats_kernel[(student_2d.shape[0],)](
            student_2d,
            teacher_2d,
            student_row_max,
            teacher_row_max,
            student_log_denom,
            teacher_log_denom,
            local_loss,
            student_2d.shape[0],
            student_2d.shape[1],
            student_2d.stride(0),
            teacher_2d.stride(0),
            local_loss.stride(0),
            1.0 / temperature,
            reverse=reverse,
        )

        ctx.input_shape = student_logits.shape
        ctx.temperature = temperature
        ctx.reverse = reverse
        ctx.tp_group = tp_group
        teacher_cache = _build_teacher_cache_triton(
            teacher_2d,
            teacher_row_max,
            teacher_log_denom,
            temperature=temperature,
            store_exp=not reverse,
        )
        ctx.save_for_backward(student_2d, teacher_cache, student_row_max, student_log_denom)
        return local_loss.reshape(leading_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        student_2d, teacher_cache, student_row_max, student_log_denom = ctx.saved_tensors

        grad_output_2d = grad_output.reshape(-1).contiguous().float()
        grad_student = torch.empty_like(student_2d, dtype=torch.float32)
        num_rows, num_cols = student_2d.shape[0], student_2d.shape[1]

        if ctx.reverse:
            local_reverse_kl = torch.empty(
                num_rows, device=student_2d.device, dtype=torch.float32
            )
            _grouped_reverse_kl_local_kernel[(num_rows,)](
                student_2d,
                teacher_cache,
                student_row_max,
                student_log_denom,
                local_reverse_kl,
                num_rows,
                num_cols,
                student_2d.stride(0),
                teacher_cache.stride(0),
                local_reverse_kl.stride(0),
                1.0 / ctx.temperature,
            )
            global_loss = local_reverse_kl.clone()
            if _tp_world_size(ctx.tp_group) > 1:
                dist.all_reduce(global_loss, op=dist.ReduceOp.SUM, group=ctx.tp_group)

            grid = lambda meta: (num_rows, triton.cdiv(num_cols, meta["BLOCK_SIZE_V"]))
            _grouped_kl_backward_kernel[grid](
                student_2d,
                teacher_cache,
                student_row_max,
                student_log_denom,
                global_loss,
                grad_output_2d,
                grad_student,
                num_rows,
                num_cols,
                student_2d.stride(0),
                teacher_cache.stride(0),
                grad_student.stride(0),
                1.0 / ctx.temperature,
            )
        else:
            grid = lambda meta: (num_rows, triton.cdiv(num_cols, meta["BLOCK_SIZE_V"]))
            _grouped_kl_backward_forward_kernel[grid](
                student_2d,
                teacher_cache,
                student_row_max,
                student_log_denom,
                grad_output_2d,
                grad_student,
                num_rows,
                num_cols,
                student_2d.stride(0),
                teacher_cache.stride(0),
                grad_student.stride(0),
                1.0 / ctx.temperature,
            )

        grad_student = grad_student.reshape(ctx.input_shape).to(student_2d.dtype)
        return grad_student, None, None, None, None


def triton_grouped_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    reverse: bool = False,
) -> torch.Tensor:
    """Forward-only Triton grouped KL benchmark kernel.

    The function expects CUDA tensors and returns per-token KL with shape
    `student_logits.shape[:-1]`.
    """

    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in the current environment.")

    student_2d, teacher_2d, leading_shape = _validate_inputs(
        student_logits, teacher_logits, temperature
    )

    if student_2d.device.type != "cuda" or teacher_2d.device.type != "cuda":
        raise ValueError("triton_grouped_kl expects CUDA tensors.")
    if student_2d.device != teacher_2d.device:
        raise ValueError("student and teacher must be on the same CUDA device.")

    output = torch.empty(student_2d.shape[0], device=student_2d.device, dtype=torch.float32)
    grid = (student_2d.shape[0],)

    _grouped_kl_kernel[grid](
        student_2d,
        teacher_2d,
        output,
        student_2d.shape[0],
        student_2d.shape[1],
        student_2d.stride(0),
        teacher_2d.stride(0),
        output.stride(0),
        1.0 / temperature,
        reverse=reverse,
    )

    return output.reshape(leading_shape)


def triton_grouped_kl_tp(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    reverse: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """TP-aware Triton grouped KL with custom backward.

    The returned tensor contains the local shard contribution with shape
    `student_logits.shape[:-1]`. During backward, the custom autograd function
    reconstructs the full TP-consistent student gradient using saved global row
    statistics and the globally summed KL row values.
    """

    return TritonGroupedKLFunction.apply(
        student_logits,
        teacher_logits,
        temperature,
        reverse,
        tp_group,
    )
