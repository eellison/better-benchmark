"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: f31e2992b351
Shape hash: dea79e0d
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
_shapes_config = "(T([], i64), T([], i64))"

class Repro(torch.nn.Module):
    def forward(self, full_default: "i64[]", full_default_1: "i64[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:167 in _get_least_common_mult_chunk_len, code: return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 64, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_default: "i64[]" = torch.ops.prims.div.default(full_default, full_default_2);  full_default = full_default_2 = None
        mul_tensor: "i64[]" = torch.ops.aten.mul.Tensor(div_default, full_default_1);  div_default = full_default_1 = None
        abs_default: "i64[]" = torch.ops.aten.abs.default(mul_tensor);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:2013 in forward, code: input_shape[-1] % least_common_mult_chunk_length != 0
        remainder_scalar_tensor: "i64[]" = torch.ops.aten.remainder.Scalar_Tensor(4096, abs_default);  abs_default = None
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        eq_tensor: "b8[]" = torch.ops.aten.eq.Tensor(remainder_scalar_tensor, full_default_3);  remainder_scalar_tensor = full_default_3 = None
        bitwise_not_default: "b8[]" = torch.ops.aten.bitwise_not.default(eq_tensor);  eq_tensor = None
        return bitwise_not_default



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
