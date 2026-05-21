"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: 088b82ab0426
Shape hash: 7f6b6d2f
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
_shapes_config = "(T([6144, 50304], bf16))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[6144, 50304]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:82 in impl, code: grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        div_tensor: "bf16[6144, 50304]" = torch.ops.aten.div.Tensor(arg0_1, 0.002232142857142857);  arg0_1 = None
        convert_element_type_default: "f8e5m2[6144, 50304]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float8_e5m2);  div_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:94 in impl, code: grad_f8.T.contiguous().T,
        permute_default: "f8e5m2[50304, 6144]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        clone_default: "f8e5m2[50304, 6144]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        permute_default_1: "f8e5m2[6144, 50304]" = torch.ops.aten.permute.default(clone_default, [1, 0]);  clone_default = None
        return permute_default_1



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
