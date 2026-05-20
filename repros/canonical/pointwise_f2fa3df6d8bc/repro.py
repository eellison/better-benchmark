"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: f2fa3df6d8bc
Shape hash: e817ec8c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 1024], bf16), T([16, 64], bf16), T([16, 64], bf16), T([512, 1024], bf16), T([512, 1, 1024], bf16), T([1024, 1024], bf16, stride=(3072, 1)), S([512, 1, 1, 16, 64]), S([512, 1, 16, 64]), S([16, 512, 64]), S([16, 512, 64]), S([512, 1, 1, 16, 64]), S([512, 1, 16, 64]), S([16, 64, 512]), S([1, 512, 1024]), S([1024, 1, 1, 16, 64]), S([1024, 1, 16, 64]), S([16, 64, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_12: "bf16[512, 1024]", _param_constant9: "bf16[16, 64]", _param_constant10: "bf16[16, 64]", mm_13: "bf16[512, 1024]", convert_element_type_10: "bf16[512, 1, 1024]", getitem_23: "bf16[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        unsqueeze_default: "bf16[1, 512, 1024]" = torch.ops.aten.unsqueeze.default(mm_12, 0);  mm_12 = None
        view_default: "bf16[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default, [0, 2, 3, 4, 1]);  view_default = None
        view_default_1: "bf16[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        add_tensor: "bf16[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_default_1, _param_constant9);  _param_constant9 = None
        unsqueeze_default_1: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, 4);  add_tensor = None
        permute_default_1: "bf16[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_1, [1, 2, 0, 4, 3]);  unsqueeze_default_1 = None
        permute_default_2: "bf16[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_1, [1, 2, 4, 0, 3]);  permute_default_1 = None
        view_default_2: "bf16[16, 512, 64]" = torch.ops.aten.view.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None
        add_tensor_1: "bf16[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_default_1, _param_constant10);  view_default_1 = _param_constant10 = None
        unsqueeze_default_2: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 4);  add_tensor_1 = None
        permute_default_3: "bf16[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_2, [1, 2, 0, 4, 3]);  unsqueeze_default_2 = None
        permute_default_4: "bf16[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_3, [1, 2, 4, 0, 3]);  permute_default_3 = None
        view_default_3: "bf16[16, 512, 64]" = torch.ops.aten.view.default(permute_default_4, _shape_param_3);  permute_default_4 = _shape_param_3 = None
        unsqueeze_default_3: "bf16[1, 512, 1024]" = torch.ops.aten.unsqueeze.default(mm_13, 0);  mm_13 = None
        view_default_4: "bf16[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default_3, _shape_param_4);  unsqueeze_default_3 = _shape_param_4 = None
        permute_default_5: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 3, 4, 1]);  view_default_4 = None
        view_default_5: "bf16[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None
        unsqueeze_default_4: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default_5, 4);  view_default_5 = None
        permute_default_6: "bf16[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_default_4, [1, 2, 4, 0, 3]);  unsqueeze_default_4 = None
        permute_default_7: "bf16[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_default_6, [1, 4, 0, 3, 2]);  permute_default_6 = None
        view_default_6: "bf16[16, 64, 512]" = torch.ops.aten.view.default(permute_default_7, _shape_param_6);  permute_default_7 = _shape_param_6 = None
        unsqueeze_default_5: "bf16[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, 3);  convert_element_type_10 = None
        unsqueeze_default_6: "bf16[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 4);  unsqueeze_default_5 = None
        permute_default_8: "bf16[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_6, [0, 1, 3, 4, 2]);  unsqueeze_default_6 = None
        permute_default_9: "bf16[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_default_8, [0, 4, 1, 2, 3]);  permute_default_8 = None
        view_default_7: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(permute_default_9, _shape_param_7);  permute_default_9 = _shape_param_7 = None
        squeeze_dim: "bf16[512, 1024]" = torch.ops.aten.squeeze.dim(view_default_7, 0);  view_default_7 = None
        unsqueeze_default_7: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(getitem_23, 0);  getitem_23 = None
        view_default_8: "bf16[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default_7, _shape_param_8);  unsqueeze_default_7 = _shape_param_8 = None
        permute_default_10: "bf16[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default_8, [0, 2, 3, 4, 1]);  view_default_8 = None
        view_default_9: "bf16[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_default_10, _shape_param_9);  permute_default_10 = _shape_param_9 = None
        unsqueeze_default_8: "bf16[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default_9, 4);  view_default_9 = None
        permute_default_11: "bf16[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_default_8, [1, 2, 4, 0, 3]);  unsqueeze_default_8 = None
        permute_default_12: "bf16[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_default_11, [1, 4, 0, 3, 2]);  permute_default_11 = None
        view_default_10: "bf16[16, 64, 1024]" = torch.ops.aten.view.default(permute_default_12, _shape_param_10);  permute_default_12 = _shape_param_10 = None
        return (view_default_2, view_default_3, view_default_6, squeeze_dim, view_default_10)


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
