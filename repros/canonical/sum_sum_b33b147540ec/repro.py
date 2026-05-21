"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: b33b147540ec
Shape hash: 34c5f94b
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
_shapes_config = "(T([1, 6, 6144, 128], bf16, stride=(4718592, 128, 768, 1)), T([1, 6, 6144, 128], bf16, stride=(4718592, 128, 768, 1)), T([1, 6, 6144, 128], bf16, stride=(4718592, 128, 768, 1)), T([64], f32), T([6144, 2304], bf16), T([262144, 64], f32), T([262144, 64], f32), T([1, 6144, 6, 1], f32), T([1, 6144, 6, 1], f32), S([-1, 2]), S([1, 6144, 2304]), S([1, 6144, 18, 128]), S([1, 6144, 2304]), S([6144, 2304]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_129: "bf16[1, 6, 6144, 128]", getitem_128: "bf16[1, 6, 6144, 128]", getitem_127: "bf16[1, 6, 6144, 128]", primals_7: "f32[64]", mm_52: "bf16[6144, 2304]", primals_71: "f32[262144, 64]", primals_72: "f32[262144, 64]", rsqrt_44: "f32[1, 6144, 6, 1]", rsqrt_43: "f32[1, 6144, 6, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_default: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_129, [0, 2, 1, 3]);  getitem_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_default_1: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3]);  getitem_128 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_default_2: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_127, [0, 2, 1, 3]);  getitem_127 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:915 in forward, code: sa_lambdas = self.scalars[3 * len(self.blocks) : 5 * len(self.blocks)].view(
        slice_tensor: "f32[24]" = torch.ops.aten.slice.Tensor(primals_7, 0, 36, 60);  primals_7 = None
        reshape_default: "f32[12, 2]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_int: "f32[2]" = torch.ops.aten.select.int(reshape_default, 0, 11);  reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_int_1: "f32[]" = torch.ops.aten.select.int(select_int, 0, 0);  select_int = None
        mul_tensor: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_default, select_int_1);  permute_default = select_int_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        reshape_default_1: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_52, _shape_param_1);  mm_52 = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        reshape_default_2: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default_2, 6, -2);  reshape_default_2 = None
        getitem: "bf16[1, 6144, 6, 128]" = split_tensor[2]
        getitem_130: "bf16[1, 6144, 6, 128]" = split_tensor[1]
        getitem_131: "bf16[1, 6144, 6, 128]" = split_tensor[0];  split_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_default: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_default_1, torch.float32);  permute_default_1 = None
        slice_tensor_1: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_default, 3, 0, 64)
        slice_tensor_2: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_default, 3, 64, 128);  convert_element_type_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_71, 0);  primals_71 = None
        slice_tensor_3: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default, 1, 0, 6144);  unsqueeze_default = None
        unsqueeze_default_1: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor_3, 2);  slice_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_tensor_1: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_2, unsqueeze_default_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default_2: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(primals_72, 0);  primals_72 = None
        slice_tensor_4: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default_2, 1, 0, 6144);  unsqueeze_default_2 = None
        unsqueeze_default_3: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor_4, 2);  slice_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_default: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_default_3)
        mul_tensor_2: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_2, neg_default);  slice_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_3: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_1, unsqueeze_default_3)
        add_tensor: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = mul_tensor_3 = None
        mul_tensor_4: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_1, unsqueeze_default_1);  slice_tensor_1 = None
        add_tensor_1: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_default: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_tensor_1, add_tensor], 3);  add_tensor_1 = add_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        convert_element_type_default_1: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(permute_default_2, torch.float32);  permute_default_2 = None
        slice_tensor_5: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_default_1, 3, 0, 64)
        slice_tensor_6: "f32[1, 6144, 6, 64]" = torch.ops.aten.slice.Tensor(convert_element_type_default_1, 3, 64, 128);  convert_element_type_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        mul_tensor_5: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_6, unsqueeze_default_1)
        mul_tensor_6: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_6, neg_default);  slice_tensor_6 = neg_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_7: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_5, unsqueeze_default_3);  unsqueeze_default_3 = None
        add_tensor_2: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_7);  mul_tensor_5 = mul_tensor_7 = None
        mul_tensor_8: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_5, unsqueeze_default_1);  slice_tensor_5 = unsqueeze_default_1 = None
        add_tensor_3: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_8);  mul_tensor_6 = mul_tensor_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        cat_default_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_tensor_3, add_tensor_2], 3);  add_tensor_3 = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_2: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_default, torch.float32);  cat_default = None
        convert_element_type_default_3: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_130, torch.float32);  getitem_130 = None
        mul_tensor_9: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, rsqrt_44);  convert_element_type_default_3 = None
        mul_tensor_10: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, convert_element_type_default_2)
        sum_dim_int_list: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [3], True);  mul_tensor_10 = None
        div_tensor: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_tensor_9, 128);  mul_tensor_9 = None
        mul_tensor_11: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_tensor, sum_dim_int_list);  div_tensor = sum_dim_int_list = None
        sub_tensor: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, mul_tensor_11);  convert_element_type_default_2 = mul_tensor_11 = None
        mul_tensor_12: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_44);  sub_tensor = rsqrt_44 = None
        convert_element_type_default_4: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        convert_element_type_default_5: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_default_1, torch.float32);  cat_default_1 = None
        convert_element_type_default_6: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
        mul_tensor_13: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, rsqrt_43);  convert_element_type_default_6 = None
        mul_tensor_14: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_13, convert_element_type_default_5)
        sum_dim_int_list_1: "f32[1, 6144, 6, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [3], True);  mul_tensor_14 = None
        div_tensor_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.div.Tensor(mul_tensor_13, 128);  mul_tensor_13 = None
        mul_tensor_15: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(div_tensor_1, sum_dim_int_list_1);  div_tensor_1 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_5, mul_tensor_15);  convert_element_type_default_5 = mul_tensor_15 = None
        mul_tensor_16: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_43);  sub_tensor_1 = rsqrt_43 = None
        convert_element_type_default_7: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16);  mul_tensor_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        cat_default_2: "bf16[1, 6144, 18, 128]" = torch.ops.aten.cat.default([convert_element_type_default_7, convert_element_type_default_4, mul_tensor], 2);  convert_element_type_default_7 = convert_element_type_default_4 = mul_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        reshape_default_3: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(cat_default_2, _shape_param_3);  cat_default_2 = _shape_param_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        reshape_default_4: "bf16[6144, 2304]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        return (reshape_default_4, getitem)



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
