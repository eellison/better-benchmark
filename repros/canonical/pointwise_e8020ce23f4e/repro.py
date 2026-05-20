"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: e8020ce23f4e
Shape hash: 90974ccb
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1024, 3072], bf16), T([512, 1024], bf16), T([16, 64], bf16), T([16, 64], bf16), T([512, 1024], bf16), T([512, 1, 1024], bf16), S([1024, 1, 1, 16, 64]), S([1024, 1, 16, 64]), S([16, 64, 1024]), S([512, 1, 1, 16, 64]), S([512, 1, 16, 64]), S([16, 512, 64]), S([16, 512, 64]), S([512, 1, 1, 16, 64]), S([512, 1, 16, 64]), S([16, 64, 512]), S([1, 512, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_3: "bf16[1024, 3072]", mm_64: "bf16[512, 1024]", _param_constant113: "bf16[16, 64]", _param_constant114: "bf16[16, 64]", mm_65: "bf16[512, 1024]", convert_element_type_114: "bf16[512, 1, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(mm_3, [1024, 1024, 1024], 1);  mm_3 = None
        getitem: "bf16[1024, 1024]" = split_with_sizes_default[0]
        getitem_1: "bf16[1024, 1024]" = split_with_sizes_default[1]
        getitem_2: "bf16[1024, 1024]" = split_with_sizes_default[2];  split_with_sizes_default = None
        unsqueeze_default: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(getitem_1, 0);  getitem_1 = None
        view_default: "bf16[1024, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "bf16[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default, [0, 2, 3, 4, 1]);  view_default = None
        view_default_1: "bf16[1024, 1, 16, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        unsqueeze_default_1: "bf16[1024, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default_1, 4);  view_default_1 = None
        permute_default_1: "bf16[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_default_1, [1, 2, 4, 0, 3]);  unsqueeze_default_1 = None
        permute_default_2: "bf16[16, 64, 1, 1024, 1]" = torch.ops.aten.permute.default(permute_default_1, [1, 4, 0, 3, 2]);  permute_default_1 = None
        view_default_2: "bf16[16, 64, 1024]" = torch.ops.aten.view.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None
        unsqueeze_default_2: "bf16[1, 512, 1024]" = torch.ops.aten.unsqueeze.default(mm_64, 0);  mm_64 = None
        view_default_3: "bf16[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default_2, _shape_param_3);  unsqueeze_default_2 = _shape_param_3 = None
        permute_default_3: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 3, 4, 1]);  view_default_3 = None
        view_default_4: "bf16[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_default_3, _shape_param_4);  permute_default_3 = _shape_param_4 = None
        add_tensor: "bf16[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_default_4, _param_constant113);  _param_constant113 = None
        unsqueeze_default_3: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, 4);  add_tensor = None
        permute_default_4: "bf16[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_3, [1, 2, 0, 4, 3]);  unsqueeze_default_3 = None
        permute_default_5: "bf16[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_4, [1, 2, 4, 0, 3]);  permute_default_4 = None
        view_default_5: "bf16[16, 512, 64]" = torch.ops.aten.view.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None
        add_tensor_1: "bf16[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(view_default_4, _param_constant114);  view_default_4 = _param_constant114 = None
        unsqueeze_default_4: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 4);  add_tensor_1 = None
        permute_default_6: "bf16[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_default_4, [1, 2, 0, 4, 3]);  unsqueeze_default_4 = None
        permute_default_7: "bf16[16, 512, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_6, [1, 2, 4, 0, 3]);  permute_default_6 = None
        view_default_6: "bf16[16, 512, 64]" = torch.ops.aten.view.default(permute_default_7, _shape_param_6);  permute_default_7 = _shape_param_6 = None
        unsqueeze_default_5: "bf16[1, 512, 1024]" = torch.ops.aten.unsqueeze.default(mm_65, 0);  mm_65 = None
        view_default_7: "bf16[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default_5, _shape_param_7);  unsqueeze_default_5 = _shape_param_7 = None
        permute_default_8: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default_7, [0, 2, 3, 4, 1]);  view_default_7 = None
        view_default_8: "bf16[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_default_8, _shape_param_8);  permute_default_8 = _shape_param_8 = None
        unsqueeze_default_6: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default_8, 4);  view_default_8 = None
        permute_default_9: "bf16[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_default_6, [1, 2, 4, 0, 3]);  unsqueeze_default_6 = None
        permute_default_10: "bf16[16, 64, 1, 512, 1]" = torch.ops.aten.permute.default(permute_default_9, [1, 4, 0, 3, 2]);  permute_default_9 = None
        view_default_9: "bf16[16, 64, 512]" = torch.ops.aten.view.default(permute_default_10, _shape_param_9);  permute_default_10 = _shape_param_9 = None
        unsqueeze_default_7: "bf16[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, 3);  convert_element_type_114 = None
        unsqueeze_default_8: "bf16[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 4);  unsqueeze_default_7 = None
        permute_default_11: "bf16[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_8, [0, 1, 3, 4, 2]);  unsqueeze_default_8 = None
        permute_default_12: "bf16[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_default_11, [0, 4, 1, 2, 3]);  permute_default_11 = None
        view_default_10: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(permute_default_12, _shape_param_10);  permute_default_12 = _shape_param_10 = None
        squeeze_dim: "bf16[512, 1024]" = torch.ops.aten.squeeze.dim(view_default_10, 0);  view_default_10 = None
        return (view_default_2, view_default_5, view_default_6, view_default_9, squeeze_dim, getitem, getitem_2)


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
