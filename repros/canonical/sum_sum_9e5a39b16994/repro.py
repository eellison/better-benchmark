"""
Standalone repro captured via capture_hook.
Label: torchbench_nanogpt_train
Pattern hash: 9e5a39b16994
Shape hash: 3b21f4a4
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
_shapes_config = "(T([1, 768], f32), T([1], i64, gen=Index(1)), T([768], f32), T([1, 64, 768], f32), T([1, 64, 1], f32), S([1, 1, 768]), S([64, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_1: "f32[1, 768]", full_default: "i64[1]", primals_148: "f32[768]", mul_96: "f32[1, 64, 768]", div: "f32[1, 64, 1]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:239 in forward, code: logits = self.lm_head(
        reshape_default: "f32[1, 1, 768]" = torch.ops.aten.reshape.default(mm_1, _shape_param_0);  mm_1 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:240 in forward, code: x[:, [-1], :]
        full_default_1: "f32[1, 64, 768]" = torch.ops.aten.full.default([1, 64, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1, 64, 768]" = torch.ops.aten.index_put.default(full_default_1, [None, full_default], reshape_default, True);  full_default_1 = full_default = reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        mul_tensor: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(index_put_default, primals_148);  index_put_default = primals_148 = None
        mul_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_96);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_96, sum_dim_int_list_1);  mul_96 = sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(div, sub_tensor_1);  div = sub_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:131 in forward, code: x = self.c_proj(x)
        reshape_default_1: "f32[64, 768]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        return reshape_default_1



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
