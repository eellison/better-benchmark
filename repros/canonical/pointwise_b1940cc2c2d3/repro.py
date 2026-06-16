"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer
Pattern hash: b1940cc2c2d3
Shape hash: d7517139
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
    def forward(self, _shape_param_0):
        # No stacktrace found for following nodes
        full: "f32[8, 512]" = torch.ops.aten.full.default(_shape_param_0, 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        unsqueeze: "f32[8, 512, 1]" = torch.ops.aten.unsqueeze.default(full, 2)
        unsqueeze_1: "f32[8, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1)
        unsqueeze_2: "f32[8, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2)
        squeeze: "f32[8, 1, 512]" = torch.ops.aten.squeeze.dim(unsqueeze_2, -2)
        unsqueeze_3: "f32[8, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(squeeze, -1)
        mul: "f32[8, 1, 512, 512]" = torch.ops.aten.mul.Tensor(unsqueeze_2, unsqueeze_3)
        convert_element_type: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_1: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_2: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_3: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_4: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_5: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_6: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_7: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_8: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_9: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_10: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_11: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_12: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_13: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_14: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_15: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_16: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_17: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_18: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_19: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_20: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_21: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_22: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        convert_element_type_23: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bool)
        return (full, unsqueeze, unsqueeze_1, unsqueeze_2, squeeze, unsqueeze_3, mul, convert_element_type, convert_element_type_1, convert_element_type_2, convert_element_type_3, convert_element_type_4, convert_element_type_5, convert_element_type_6, convert_element_type_7, convert_element_type_8, convert_element_type_9, convert_element_type_10, convert_element_type_11, convert_element_type_12, convert_element_type_13, convert_element_type_14, convert_element_type_15, convert_element_type_16, convert_element_type_17, convert_element_type_18, convert_element_type_19, convert_element_type_20, convert_element_type_21, convert_element_type_22, convert_element_type_23)



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
