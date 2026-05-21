"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: 56fa1ac98eb3
Shape hash: 8c80d321
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([6144, 2304], bf16), T([262144, 64], f32), T([262144, 64], f32), T([12, 2], f32), S([1, 6144, 2304]), S([1, 6144, 18, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_30: "bf16[6144, 2304]", arg52_1: "f32[262144, 64]", arg53_1: "f32[262144, 64]", view_7: "f32[12, 2]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        reshape_default: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_30, _shape_param_0);  mm_30 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        reshape_default_1: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default_1, 6, -2);  reshape_default_1 = None
        getitem: "bf16[1, 6144, 6, 128]" = split_tensor[0]
        getitem_1: "bf16[1, 6144, 6, 128]" = split_tensor[1]
        getitem_2: "bf16[1, 6144, 6, 128]" = split_tensor[2];  split_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        pow_tensor_scalar: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [3], True);  pow_tensor_scalar = None
        add_scalar: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_dim, 1.1920928955078125e-07);  mean_dim = None
        rsqrt_default: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_scalar);  add_scalar = None
        mul_tensor: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_1: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        split_tensor_1 = torch.ops.aten.split.Tensor(convert_element_type_default_1, 64, -1);  convert_element_type_default_1 = None
        getitem_3: "f32[1, 6144, 6, 64]" = split_tensor_1[0]
        getitem_4: "f32[1, 6144, 6, 64]" = split_tensor_1[1];  split_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_2: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        pow_tensor_scalar_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim_1: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [3], True);  pow_tensor_scalar_1 = None
        add_scalar_1: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_dim_1, 1.1920928955078125e-07);  mean_dim_1 = None
        rsqrt_default_1: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_scalar_1);  add_scalar_1 = None
        mul_tensor_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default_1);  convert_element_type_default_2 = rsqrt_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_3: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        split_tensor_2 = torch.ops.aten.split.Tensor(convert_element_type_default_3, 64, -1);  convert_element_type_default_3 = None
        getitem_5: "f32[1, 6144, 6, 64]" = split_tensor_2[0]
        getitem_6: "f32[1, 6144, 6, 64]" = split_tensor_2[1];  split_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg52_1, 0)
        slice_tensor: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default, 1, 0, 6144);  unsqueeze_default = None
        unsqueeze_default_1: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor, 2);  slice_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_2: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_default_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default_2: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg53_1, 0)
        slice_tensor_1: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default_2, 1, 0, 6144);  unsqueeze_default_2 = None
        unsqueeze_default_3: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor_1, 2);  slice_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_3: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_4, unsqueeze_default_3)
        add_tensor: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_default: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_default_3);  unsqueeze_default_3 = None
        mul_tensor_4: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_3, neg_default);  getitem_3 = neg_default = None
        mul_tensor_5: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_4, unsqueeze_default_1);  getitem_4 = unsqueeze_default_1 = None
        add_tensor_1: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_default: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_tensor, add_tensor_1], 3);  add_tensor = add_tensor_1 = None
        convert_element_type_default_4: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_default, torch.bfloat16);  cat_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_default: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 2, 1, 3]);  convert_element_type_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default_4: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg52_1, 0);  arg52_1 = None
        slice_tensor_2: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default_4, 1, 0, 6144);  unsqueeze_default_4 = None
        unsqueeze_default_5: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor_2, 2);  slice_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_6: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze_default_5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default_6: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg53_1, 0);  arg53_1 = None
        slice_tensor_3: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default_6, 1, 0, 6144);  unsqueeze_default_6 = None
        unsqueeze_default_7: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor_3, 2);  slice_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_7: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze_default_7)
        add_tensor_2: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_default_1: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_default_7);  unsqueeze_default_7 = None
        mul_tensor_8: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_5, neg_default_1);  getitem_5 = neg_default_1 = None
        mul_tensor_9: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze_default_5);  getitem_6 = unsqueeze_default_5 = None
        add_tensor_3: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_default_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_tensor_2, add_tensor_3], 3);  add_tensor_2 = add_tensor_3 = None
        convert_element_type_default_5: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_default_1, torch.bfloat16);  cat_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_default_1: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_default_5, [0, 2, 1, 3]);  convert_element_type_default_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_int: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 8);  view_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        select_int_1: "f32[]" = torch.ops.aten.select.int(select_int, 0, 0);  select_int = None
        mul_tensor_10: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_int_1, getitem_2);  select_int_1 = getitem_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_default_2: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(mul_tensor_10, [0, 2, 1, 3]);  mul_tensor_10 = None
        return (permute_default_2, permute_default, permute_default_1)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
