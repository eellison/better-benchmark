"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_infer
Pattern hash: f57ae4150066
Shape hash: 78b16e07
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
_shapes_config = "(T([1, 512, 768], f16), T([50265, 768], f16), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[1, 512, 768]", arg0_1: "f16[50265, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:939 in torch_dynamo_resume_in_forward_at_926, code: lm_logits = self.lm_head(outputs[0])
        reshape_default: "f16[512, 768]" = torch.ops.aten.reshape.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        permute_default: "f16[768, 50265]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        constant_pad_nd_default: "f16[768, 50272]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 7, 0, 0]);  permute_default = None
        return (reshape_default, constant_pad_nd_default)



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
