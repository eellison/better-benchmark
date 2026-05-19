"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-4-5-linux.aws.a100_graph10
Pattern hash: 33c786e4cf12
Shape hash: 3208e585
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
    def forward(self, addmm_66: "bf16[256, 1024]", addmm_67: "bf16[256, 1024]", convert_element_type_89: "bf16[1, 256, 1024]", arg185_1: "bf16[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "bf16[1, 256, 1024]" = torch.ops.aten.view.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None
        mul_tensor: "bf16[1, 256, 1024]" = torch.ops.aten.mul.Tensor(view_default, 0.125);  view_default = None
        view_default_1: "bf16[1, 256, 1024]" = torch.ops.aten.view.default(addmm_67, _shape_param_1);  addmm_67 = _shape_param_1 = None
        view_default_2: "bf16[256, 1024]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_2);  convert_element_type_89 = _shape_param_2 = None
        permute_default: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        view_default_3: "bf16[1, 256, 16, 64]" = torch.ops.aten.view.default(view_default_1, _shape_param_3);  view_default_1 = _shape_param_3 = None
        permute_default_1: "bf16[1, 16, 256, 64]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        view_default_4: "bf16[1, 256, 16, 64]" = torch.ops.aten.view.default(mul_tensor, _shape_param_4);  mul_tensor = _shape_param_4 = None
        permute_default_2: "bf16[1, 16, 256, 64]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        view_default_5: "bf16[16, 256, 64]" = torch.ops.aten.view.default(permute_default_2, _shape_param_5);  permute_default_2 = _shape_param_5 = None
        view_default_6: "bf16[16, 256, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_6);  permute_default_1 = _shape_param_6 = None
        permute_default_3: "bf16[16, 64, 256]" = torch.ops.aten.permute.default(view_default_6, [0, 2, 1]);  view_default_6 = None
        return (view_default_2, permute_default, view_default_5, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([256, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([256, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 256, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    [1, 256, 1024],  # _shape_param_0
    [1, 256, 1024],  # _shape_param_1
    [256, 1024],  # _shape_param_2
    [1, -1, 16, 64],  # _shape_param_3
    [1, 256, 16, 64],  # _shape_param_4
    [16, 256, 64],  # _shape_param_5
    [16, 256, 64],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
