"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: b09beca3a7bd
Shape hash: 26d25975
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 4096]", arg2_1: "bf16[4096, 4096]", arg3_1: "bf16[4096, 4096]", arg4_1: "bf16[4096, 4096]", arg5_1: "bf16[4096, 4096]", arg6_1: "bf16[4096, 4096]", arg7_1: "bf16[4096, 4096]", arg8_1: "bf16[4096, 4096]", arg9_1: "bf16[4096, 4096]", arg10_1: "bf16[4096, 4096]", arg11_1: "bf16[4096, 4096]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        add: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        convert_element_type_2: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add_1: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        convert_element_type_3: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add_2: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        convert_element_type_4: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        add_3: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_4);  add_2 = convert_element_type_4 = None
        convert_element_type_5: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        add_4: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_3, convert_element_type_5);  add_3 = convert_element_type_5 = None
        convert_element_type_6: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        add_5: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_6);  add_4 = convert_element_type_6 = None
        convert_element_type_7: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        add_6: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_7);  add_5 = convert_element_type_7 = None
        convert_element_type_8: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        add_7: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_8);  add_6 = convert_element_type_8 = None
        convert_element_type_9: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        add_8: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_9);  add_7 = convert_element_type_9 = None
        convert_element_type_10: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32);  arg10_1 = None
        add_9: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_8, convert_element_type_10);  add_8 = convert_element_type_10 = None
        convert_element_type_11: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
        add_10: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_9, convert_element_type_11);  add_9 = convert_element_type_11 = None
        return add_10



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
