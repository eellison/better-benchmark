"""
Standalone repro captured via capture_hook.
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([2048, 3072], bf16), T([2048, 3072], bf16), T([1024, 3072], bf16))"

class Repro(torch.nn.Module):
    def forward(self, mm_193: "bf16[2048, 3072]", mm_194: "bf16[2048, 3072]", arg310_1: "bf16[1024, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_193, [4, 512, 3072]);  mm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_default: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        neg_default: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default_1: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_194, [4, 512, 3072]);  mm_194 = None
        mul_tensor: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, reshape_default_1);  convert_element_type_default_1 = reshape_default_1 = None
        reshape_default_2: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_tensor, [2048, 3072]);  mul_tensor = None
        permute_default: "bf16[3072, 1024]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        return (reshape_default_2, permute_default)


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
