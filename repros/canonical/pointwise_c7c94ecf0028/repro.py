"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-1-5-linux.aws.a100_graph2
Pattern hash: c7c94ecf0028
Shape hash: bcc87dc5
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
    def forward(self, addmm_66: "bf16[1024, 768]", addmm_67: "bf16[1024, 768]", permute_517: "bf16[1024, 1, 768]", arg183_1: "bf16[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        view_default: "bf16[1024, 1, 768]" = torch.ops.aten.view.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None
        view_default_1: "bf16[1024, 1, 768]" = torch.ops.aten.view.default(addmm_67, _shape_param_1);  addmm_67 = _shape_param_1 = None
        view_default_2: "bf16[1024, 768]" = torch.ops.aten.view.default(permute_517, _shape_param_2);  permute_517 = _shape_param_2 = None
        permute_default: "bf16[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        div_tensor: "bf16[1024, 1, 768]" = torch.ops.aten.div.Tensor(view_default, 8.0);  view_default = None
        view_default_3: "bf16[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_default_1, _shape_param_3);  view_default_1 = _shape_param_3 = None
        permute_default_1: "bf16[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_3, [1, 0, 2, 3]);  view_default_3 = None
        permute_default_2: "bf16[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default_1, [0, 2, 1, 3]);  permute_default_1 = None
        view_default_4: "bf16[12, 1024, 64]" = torch.ops.aten.view.default(permute_default_2, _shape_param_4);  permute_default_2 = _shape_param_4 = None
        view_default_5: "bf16[12, 2, 512, 64]" = torch.ops.aten.view.default(view_default_4, _shape_param_5);  view_default_4 = _shape_param_5 = None
        as_strided_default: "bf16[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_default_5, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_default_5 = None
        unsqueeze_default: "bf16[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        permute_default_3: "bf16[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 2, 3]);  unsqueeze_default = None
        view_default_6: "bf16[1024, 1, 12, 64]" = torch.ops.aten.view.default(div_tensor, _shape_param_6);  div_tensor = _shape_param_6 = None
        permute_default_4: "bf16[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_6, [1, 0, 2, 3]);  view_default_6 = None
        permute_default_5: "bf16[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default_4, [0, 2, 1, 3]);  permute_default_4 = None
        view_default_7: "bf16[12, 1024, 64]" = torch.ops.aten.view.default(permute_default_5, _shape_param_7);  permute_default_5 = _shape_param_7 = None
        view_default_8: "bf16[12, 2, 512, 64]" = torch.ops.aten.view.default(view_default_7, _shape_param_8);  view_default_7 = _shape_param_8 = None
        as_strided_default_1: "bf16[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_default_8, [12, 3, 512, 64], [64, 196608, 768, 1]);  view_default_8 = None
        unsqueeze_default_1: "bf16[12, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default_1, 4);  as_strided_default_1 = None
        clone_default: "bf16[12, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default_1, memory_format = torch.contiguous_format);  unsqueeze_default_1 = None
        view_default_9: "bf16[36, 512, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_9);  clone_default = _shape_param_9 = None
        permute_default_6: "bf16[12, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_default_3, [0, 1, 4, 3, 2]);  permute_default_3 = None
        clone_default_1: "bf16[12, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_default_6, memory_format = torch.contiguous_format);  permute_default_6 = None
        view_default_10: "bf16[36, 64, 512]" = torch.ops.aten.view.default(clone_default_1, _shape_param_10);  clone_default_1 = _shape_param_10 = None
        return (view_default_2, permute_default, view_default_9, view_default_10)


def _default_make_inputs():
    return [
    torch.randn([1024, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768, 768], dtype=torch.bfloat16, device='cuda'),
    [1024, 1, 768],  # _shape_param_0
    [1024, 1, 768],  # _shape_param_1
    [1024, 768],  # _shape_param_2
    [1024, 1, 12, 64],  # _shape_param_3
    [12, 1024, 64],  # _shape_param_4
    [12, 2, 512, 64],  # _shape_param_5
    [1024, 1, 12, 64],  # _shape_param_6
    [12, 1024, 64],  # _shape_param_7
    [12, 2, 512, 64],  # _shape_param_8
    [36, 512, 64],  # _shape_param_9
    [36, 64, 512],  # _shape_param_10
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
