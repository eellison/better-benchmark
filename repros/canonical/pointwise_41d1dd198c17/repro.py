"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_infer
Pattern hash: 41d1dd198c17
Shape hash: db9e7031
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
_shapes_config = "(T([512, 50272], f16), T([1, 50265], f16), S([1, 512, 50265]))"

class Repro(torch.nn.Module):
    def forward(self, mm_default: "f16[512, 50272]", arg2_1: "f16[1, 50265]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:939 in torch_dynamo_resume_in_forward_at_926, code: lm_logits = self.lm_head(outputs[0])
        slice_tensor: "f16[512, 50265]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -7);  mm_default = None
        reshape_default: "f16[1, 512, 50265]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:940 in torch_dynamo_resume_in_forward_at_926, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        add_tensor: "f16[1, 512, 50265]" = torch.ops.aten.add.Tensor(reshape_default, arg2_1);  reshape_default = arg2_1 = None
        return add_tensor



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
