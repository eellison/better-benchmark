"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: 247c27751765
Shape hash: 866e46e6
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg140_1: "f32[2304]", mm_11: "f16[1600, 2304]", addmm_80: "f16[2464, 1536]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[2304]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float16);  arg140_1 = None
        reshape_default: "f16[50, 32, 2304]" = torch.ops.aten.reshape.default(mm_11, _shape_param_0);  mm_11 = _shape_param_0 = None
        add_tensor: "f16[50, 32, 2304]" = torch.ops.aten.add.Tensor(reshape_default, convert_element_type_default);  reshape_default = convert_element_type_default = None
        reshape_default_1: "f16[50, 32, 3, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        unsqueeze_default: "f16[1, 50, 32, 3, 768]" = torch.ops.aten.unsqueeze.default(reshape_default_1, 0);  reshape_default_1 = None
        permute_default: "f16[3, 50, 32, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_default, [3, 1, 2, 0, 4]);  unsqueeze_default = None
        squeeze_dim: "f16[3, 50, 32, 768]" = torch.ops.aten.squeeze.dim(permute_default, -2);  permute_default = None
        clone_default: "f16[3, 50, 32, 768]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        select_int: "f16[50, 32, 768]" = torch.ops.aten.select.int(clone_default, 0, 0)
        select_int_1: "f16[50, 32, 768]" = torch.ops.aten.select.int(clone_default, 0, 1)
        select_int_2: "f16[50, 32, 768]" = torch.ops.aten.select.int(clone_default, 0, 2);  clone_default = None
        reshape_default_2: "f16[50, 384, 64]" = torch.ops.aten.reshape.default(select_int, _shape_param_2);  select_int = _shape_param_2 = None
        permute_default_1: "f16[384, 50, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0, 2]);  reshape_default_2 = None
        reshape_default_3: "f16[50, 384, 64]" = torch.ops.aten.reshape.default(select_int_1, _shape_param_3);  select_int_1 = _shape_param_3 = None
        permute_default_2: "f16[384, 50, 64]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0, 2]);  reshape_default_3 = None
        reshape_default_4: "f16[50, 384, 64]" = torch.ops.aten.reshape.default(select_int_2, _shape_param_4);  select_int_2 = _shape_param_4 = None
        permute_default_3: "f16[384, 50, 64]" = torch.ops.aten.permute.default(reshape_default_4, [1, 0, 2]);  reshape_default_4 = None
        reshape_default_5: "f16[32, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_5);  permute_default_1 = _shape_param_5 = None
        reshape_default_6: "f16[32, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_6);  permute_default_2 = _shape_param_6 = None
        reshape_default_7: "f16[32, 12, 50, 64]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_7);  permute_default_3 = _shape_param_7 = None
        reshape_default_8: "f16[77, 32, 1536]" = torch.ops.aten.reshape.default(addmm_80, _shape_param_8);  addmm_80 = _shape_param_8 = None
        reshape_default_9: "f16[77, 32, 3, 512]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_9);  reshape_default_8 = _shape_param_9 = None
        unsqueeze_default_1: "f16[1, 77, 32, 3, 512]" = torch.ops.aten.unsqueeze.default(reshape_default_9, 0);  reshape_default_9 = None
        permute_default_4: "f16[3, 77, 32, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_default_1, [3, 1, 2, 0, 4]);  unsqueeze_default_1 = None
        squeeze_dim_1: "f16[3, 77, 32, 512]" = torch.ops.aten.squeeze.dim(permute_default_4, -2);  permute_default_4 = None
        clone_default_1: "f16[3, 77, 32, 512]" = torch.ops.aten.clone.default(squeeze_dim_1, memory_format = torch.contiguous_format);  squeeze_dim_1 = None
        select_int_3: "f16[77, 32, 512]" = torch.ops.aten.select.int(clone_default_1, 0, 0)
        select_int_4: "f16[77, 32, 512]" = torch.ops.aten.select.int(clone_default_1, 0, 1)
        select_int_5: "f16[77, 32, 512]" = torch.ops.aten.select.int(clone_default_1, 0, 2);  clone_default_1 = None
        reshape_default_10: "f16[77, 256, 64]" = torch.ops.aten.reshape.default(select_int_3, _shape_param_10);  select_int_3 = _shape_param_10 = None
        permute_default_5: "f16[256, 77, 64]" = torch.ops.aten.permute.default(reshape_default_10, [1, 0, 2]);  reshape_default_10 = None
        reshape_default_11: "f16[77, 256, 64]" = torch.ops.aten.reshape.default(select_int_4, _shape_param_11);  select_int_4 = _shape_param_11 = None
        permute_default_6: "f16[256, 77, 64]" = torch.ops.aten.permute.default(reshape_default_11, [1, 0, 2]);  reshape_default_11 = None
        reshape_default_12: "f16[77, 256, 64]" = torch.ops.aten.reshape.default(select_int_5, _shape_param_12);  select_int_5 = _shape_param_12 = None
        permute_default_7: "f16[256, 77, 64]" = torch.ops.aten.permute.default(reshape_default_12, [1, 0, 2]);  reshape_default_12 = None
        reshape_default_13: "f16[32, 8, 77, 64]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_13);  permute_default_5 = _shape_param_13 = None
        reshape_default_14: "f16[32, 8, 77, 64]" = torch.ops.aten.reshape.default(permute_default_6, _shape_param_14);  permute_default_6 = _shape_param_14 = None
        reshape_default_15: "f16[32, 8, 77, 64]" = torch.ops.aten.reshape.default(permute_default_7, _shape_param_15);  permute_default_7 = _shape_param_15 = None
        return (reshape_default_5, reshape_default_6, reshape_default_7, reshape_default_13, reshape_default_14, reshape_default_15)


def _default_make_inputs():
    return [
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    [50, 32, 2304],  # _shape_param_0
    [50, 32, 3, 768],  # _shape_param_1
    [50, 384, 64],  # _shape_param_2
    [50, 384, 64],  # _shape_param_3
    [50, 384, 64],  # _shape_param_4
    [32, 12, 50, 64],  # _shape_param_5
    [32, 12, 50, 64],  # _shape_param_6
    [32, 12, 50, 64],  # _shape_param_7
    [77, 32, 1536],  # _shape_param_8
    [77, 32, 3, 512],  # _shape_param_9
    [77, 256, 64],  # _shape_param_10
    [77, 256, 64],  # _shape_param_11
    [77, 256, 64],  # _shape_param_12
    [32, 8, 77, 64],  # _shape_param_13
    [32, 8, 77, 64],  # _shape_param_14
    [32, 8, 77, 64],  # _shape_param_15
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
