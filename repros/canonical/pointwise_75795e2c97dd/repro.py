"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer
Pattern hash: 75795e2c97dd
Shape hash: 5d9a9b51
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([64, 128], i64, max=128))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "i64[64, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:177 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne_scalar: "b8[64, 128]" = torch.ops.aten.ne.Scalar(arg1_1, 1);  arg1_1 = None
        convert_element_type_default: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        return convert_element_type_default


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
