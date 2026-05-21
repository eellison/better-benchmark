"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: dfd1a208fefb
Shape hash: 1ce304a7
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
_shapes_config = "(T([6144], i64), T([], f32), T([1, 6144, 50304], bf16), T([6144, 1], f32), T([6144, 1], f32), S([1, 50304]), S([6144, 50304]), S([-1, 50304]), S([1, 6144, 50304]), S([6144, 50304]))"

class Repro(torch.nn.Module):
    def forward(self, primals_77: "i64[6144]", tangents_1: "f32[]", view_184: "bf16[1, 6144, 50304]", amax: "f32[6144, 1]", log: "f32[6144, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_default: "i64[6144, 1]" = torch.ops.aten.unsqueeze.default(primals_77, 1);  primals_77 = None
        ne_scalar: "b8[6144, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        where_self: "f32[6144, 1]" = torch.ops.aten.where.self(ne_scalar, tangents_1, full_default);  tangents_1 = full_default = None
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "i64[6144, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default_1);  ne_scalar = unsqueeze_default = full_default_1 = None

        # No stacktrace found for following nodes
        iota_default: "i64[50304]" = torch.ops.prims.iota.default(50304, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default: "i64[1, 50304]" = torch.ops.aten.reshape.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None
        expand_default: "i64[6144, 50304]" = torch.ops.aten.expand.default(where_self_1, _shape_param_1);  where_self_1 = _shape_param_1 = None
        eq_tensor: "b8[6144, 50304]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default);  expand_default = reshape_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        where_self_2: "f32[6144, 50304]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        mul_tensor: "f32[6144, 50304]" = torch.ops.aten.mul.Tensor(where_self_2, where_self);  where_self_2 = where_self = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:929 in forward, code: logits = self.lm_head(x).float()
        convert_element_type_default: "f32[1, 6144, 50304]" = torch.ops.prims.convert_element_type.default(view_184, torch.float32);  view_184 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:931 in forward, code: logits = 30 * torch.sigmoid(logits / 7.5)
        div_tensor: "f32[1, 6144, 50304]" = torch.ops.aten.div.Tensor(convert_element_type_default, 7.5);  convert_element_type_default = None
        sigmoid_default: "f32[1, 6144, 50304]" = torch.ops.aten.sigmoid.default(div_tensor);  div_tensor = None
        mul_tensor_1: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(sigmoid_default, 30)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:933 in forward, code: logits.view(-1, logits.size(-1)),
        reshape_default_1: "f32[6144, 50304]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_2);  mul_tensor_1 = _shape_param_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:932 in forward, code: loss = F.cross_entropy(
        sub_tensor: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax);  reshape_default_1 = amax = None
        sub_tensor_1: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(sub_tensor, log);  sub_tensor = log = None
        exp_default: "f32[6144, 50304]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_2: "f32[6144, 50304]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[6144, 50304]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = mul_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:933 in forward, code: logits.view(-1, logits.size(-1)),
        reshape_default_2: "f32[1, 6144, 50304]" = torch.ops.aten.reshape.default(sub_tensor_2, _shape_param_3);  sub_tensor_2 = _shape_param_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:931 in forward, code: logits = 30 * torch.sigmoid(logits / 7.5)
        mul_tensor_3: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(reshape_default_2, 30);  reshape_default_2 = None
        sub_tensor_3: "f32[1, 6144, 50304]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_4: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_3);  sigmoid_default = sub_tensor_3 = None
        mul_tensor_5: "f32[1, 6144, 50304]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        div_tensor_1: "f32[1, 6144, 50304]" = torch.ops.aten.div.Tensor(mul_tensor_5, 7.5);  mul_tensor_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:929 in forward, code: logits = self.lm_head(x).float()
        convert_element_type_default_1: "bf16[1, 6144, 50304]" = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.bfloat16);  div_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:616 in forward, code: return out.reshape(*x.shape[:-1], -1)
        reshape_default_3: "bf16[6144, 50304]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_4);  convert_element_type_default_1 = _shape_param_4 = None
        return reshape_default_3



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
