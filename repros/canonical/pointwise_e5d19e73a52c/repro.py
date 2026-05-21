"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer_005
Pattern hash: e5d19e73a52c
Shape hash: ee32fdb4
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
_shapes_config = "(T([], i64))"

class Repro(torch.nn.Module):
    def forward(self, _tensor_constant5: "i64[]"):
        # No stacktrace found for following nodes
        lift_fresh_copy_default: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        convert_element_type_default: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy_default, torch.float64);  lift_fresh_copy_default = None
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
