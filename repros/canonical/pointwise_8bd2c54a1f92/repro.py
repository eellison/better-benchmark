"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-1-9-linux.aws.h100_graph62
Pattern hash: 8bd2c54a1f92
Shape hash: 700ca791
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_3: "bf16[10000, 64]", arg1_1: "i64[2, 200000]", arg12_1: "bf16[1]", arg13_1: "bf16[64, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        relu_default: "bf16[10000, 64]" = torch.ops.aten.relu.default(addmm_3);  addmm_3 = None
        select_int: "i64[200000]" = torch.ops.aten.select.int(arg1_1, 0, 0)
        index_tensor: "bf16[200000, 64]" = torch.ops.aten.index.Tensor(relu_default, [select_int]);  select_int = None
        select_int_1: "i64[200000]" = torch.ops.aten.select.int(arg1_1, 0, 1);  arg1_1 = None
        view_default: "i64[200000, 1]" = torch.ops.aten.view.default(select_int_1, [-1, 1]);  select_int_1 = None
        expand_default: "i64[200000, 64]" = torch.ops.aten.expand.default(view_default, _shape_param_0);  view_default = _shape_param_0 = None
        full_default: "bf16[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add_default: "bf16[10000, 64]" = torch.ops.aten.scatter_add.default(full_default, 0, expand_default, index_tensor);  full_default = expand_default = index_tensor = None
        add_tensor: "bf16[1]" = torch.ops.aten.add.Tensor(arg12_1, 1);  arg12_1 = None
        mul_tensor: "bf16[10000, 64]" = torch.ops.aten.mul.Tensor(add_tensor, relu_default);  add_tensor = relu_default = None
        add_tensor_1: "bf16[10000, 64]" = torch.ops.aten.add.Tensor(scatter_add_default, mul_tensor);  scatter_add_default = mul_tensor = None
        permute_default: "bf16[64, 64]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        return (add_tensor_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([10000, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 200000, [2, 200000], dtype=torch.int64, device='cuda'),
    torch.randn([1], dtype=torch.bfloat16, device='cuda'),
    torch.randn([64, 64], dtype=torch.bfloat16, device='cuda'),
    [200000, 64],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
