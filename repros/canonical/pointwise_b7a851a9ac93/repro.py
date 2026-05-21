"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer_000
Pattern hash: b7a851a9ac93
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

_repro_version = 2
_shapes_config = "()"

class Repro(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        full_default: "f32[8, 512]" = torch.ops.aten.full.default([8, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_default: "f32[8, 1, 512]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[8, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        squeeze_dim: "f32[8, 1, 512]" = torch.ops.aten.squeeze.dim(unsqueeze_default_1, -2)
        unsqueeze_default_2: "f32[8, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(squeeze_dim, -1);  squeeze_dim = None
        mul_tensor: "f32[8, 1, 512, 512]" = torch.ops.aten.mul.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None
        convert_element_type_default: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_1: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_2: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_3: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_4: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_5: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_6: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_7: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_8: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_9: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_10: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_11: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_12: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_13: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_14: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_15: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_16: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_17: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_18: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_19: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_20: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_21: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_22: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool)
        convert_element_type_default_23: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bool);  mul_tensor = None
        return (convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_7, convert_element_type_default_8, convert_element_type_default_9, convert_element_type_default_10, convert_element_type_default_11, convert_element_type_default_12, convert_element_type_default_13, convert_element_type_default_14, convert_element_type_default_15, convert_element_type_default_16, convert_element_type_default_17, convert_element_type_default_18, convert_element_type_default_19, convert_element_type_default_20, convert_element_type_default_21, convert_element_type_default_22, convert_element_type_default_23)



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
