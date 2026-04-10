from __future__ import annotations

import unittest

import torch

try:
    from .triton_grouped_kl import (
        HAS_TRITON,
        torch_reference_grouped_kl,
        torch_reference_grouped_kl_tp,
        triton_grouped_kl,
        triton_grouped_kl_tp,
    )
except ImportError:
    from triton_grouped_kl import (
        HAS_TRITON,
        torch_reference_grouped_kl,
        torch_reference_grouped_kl_tp,
        triton_grouped_kl,
        triton_grouped_kl_tp,
    )
class TritonGroupedKLTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton tests.")
    @unittest.skipUnless(HAS_TRITON, "Triton is required for Triton tests.")
    def test_matches_reference_across_shapes_dtypes_and_directions(self):
        device = torch.device("cuda")
        cases = [
            ((17, 3, 257), torch.float32, 1e-5, 1e-5),
            ((33, 2, 1024), torch.float16, 7e-3, 7e-3),
            ((65, 1, 4097), torch.bfloat16, 2e-2, 2e-2),
        ]

        for reverse in (False, True):
            for shape, dtype, atol, rtol in cases:
                with self.subTest(shape=shape, dtype=dtype, reverse=reverse):
                    torch.manual_seed(1234)
                    student = torch.randn(shape, device=device, dtype=dtype)
                    teacher = torch.randn(shape, device=device, dtype=dtype)

                    reference = torch_reference_grouped_kl(
                        student, teacher, temperature=1.0, reverse=reverse
                    )
                    actual = triton_grouped_kl(
                        student, teacher, temperature=1.0, reverse=reverse
                    )
                    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton tests.")
    @unittest.skipUnless(HAS_TRITON, "Triton is required for Triton tests.")
    def test_preserves_leading_shape(self):
        student = torch.randn((9, 4, 513), device="cuda", dtype=torch.bfloat16)
        teacher = torch.randn_like(student)
        output = triton_grouped_kl(student, teacher, temperature=0.7, reverse=False)
        self.assertEqual(output.shape, student.shape[:-1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton tests.")
    @unittest.skipUnless(HAS_TRITON, "Triton is required for Triton tests.")
    def test_tp_autograd_matches_reference_on_single_gpu(self):
        device = torch.device("cuda")
        cases = [
            ((19, 2, 257), torch.float32, 1e-5, 1e-5),
            ((37, 1, 1024), torch.float16, 7e-3, 7e-3),
            ((65, 2, 4097), torch.bfloat16, 2e-2, 2e-2),
        ]

        for reverse in (False, True):
            for shape, dtype, atol, rtol in cases:
                with self.subTest(shape=shape, dtype=dtype, reverse=reverse):
                    torch.manual_seed(5678)
                    reference_student = torch.randn(
                        shape, device=device, dtype=dtype, requires_grad=True
                    )
                    triton_student = reference_student.detach().clone().requires_grad_(True)
                    teacher = torch.randn(shape, device=device, dtype=dtype)
                    grad_weight = torch.randn(shape[:-1], device=device, dtype=torch.float32)

                    reference_loss = torch_reference_grouped_kl_tp(
                        reference_student,
                        teacher,
                        temperature=0.9,
                        reverse=reverse,
                        tp_group=None,
                    )
                    triton_loss = triton_grouped_kl_tp(
                        triton_student,
                        teacher,
                        temperature=0.9,
                        reverse=reverse,
                        tp_group=None,
                    )
                    torch.testing.assert_close(triton_loss, reference_loss, atol=atol, rtol=rtol)

                    (reference_loss.float() * grad_weight).sum().backward()
                    (triton_loss.float() * grad_weight).sum().backward()
                    torch.testing.assert_close(
                        triton_student.grad,
                        reference_student.grad,
                        atol=atol,
                        rtol=rtol,
                    )


if __name__ == "__main__":
    unittest.main()
