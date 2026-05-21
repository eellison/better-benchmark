"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: 3dc00c79dc7f
Shape hash: e17a260d
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
_shapes_config = "(T([6144], i64, gen=Index(50304)), T([6144, 50304], bf16), S([1, 6144, 50304]), S([-1, 50304]))"

class Repro(torch.nn.Module):
    def forward(self, arg76_1: "i64[6144]", mm_46: "bf16[6144, 50304]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default: "i32[1, 1]" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "i32[1, 1]" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "i32[1, 1]" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "i32[1, 1]" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        ne_scalar: "b8[6144]" = torch.ops.aten.ne.Scalar(arg76_1, -100)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        reshape_default: "bf16[1, 6144, 50304]" = torch.ops.aten.reshape.default(mm_46, _shape_param_0);  mm_46 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:929 in forward, code: logits = self.lm_head(x).float()
        convert_element_type_default: "f32[1, 6144, 50304]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:931 in forward, code: logits = 30 * torch.sigmoid(logits / 7.5)
        div_tensor: "f32[1, 6144, 50304]" = torch.ops.aten.div.Tensor(convert_element_type_default, 7.5);  convert_element_type_default = None
        sigmoid_default: "f32[1, 6144, 50304]" = torch.ops.aten.sigmoid.default(div_tensor);  div_tensor = None
        mul_tensor: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(sigmoid_default, 30);  sigmoid_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:933 in forward, code: logits.view(-1, logits.size(-1)),
        reshape_default_1: "f32[6144, 50304]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        amax_default: "f32[6144, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[6144, 50304]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[6144, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[6144, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_1: "b8[6144]" = torch.ops.aten.ne.Scalar(arg76_1, -100)
        full_default_4: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[6144]" = torch.ops.aten.where.self(ne_scalar_1, arg76_1, full_default_4);  ne_scalar_1 = full_default_4 = None
        unsqueeze_default: "i64[6144, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[6144, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[6144]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[6144]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[6144]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_5);  ne_scalar = neg_default = full_default_5 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        ne_scalar_2: "b8[6144]" = torch.ops.aten.ne.Scalar(arg76_1, -100);  arg76_1 = None
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(ne_scalar_2);  ne_scalar_2 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor_1: "f32[]" = torch.ops.aten.div.Tensor(sum_default, convert_element_type_default_1);  sum_default = convert_element_type_default_1 = None
        return (full_default, full_default_1, full_default_2, full_default_3, div_tensor_1)



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
