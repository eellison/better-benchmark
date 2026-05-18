"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g40
Pattern hash: 25dfd1c825e5
Shape hash: 17127cc2
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 474]", arg1_1: "i64[2225]"):
        # No stacktrace found for following nodes
        ne_scalar: "b8[4, 474]" = torch.ops.aten.ne.Scalar(arg0_1, 0)
        sum_dim_int_list: "i64[4]" = torch.ops.aten.sum.dim_IntList(ne_scalar, [-1])
        index_tensor: "i64[4, 474]" = torch.ops.aten.index.Tensor(arg1_1, [arg0_1]);  arg1_1 = arg0_1 = None
        eq_scalar: "b8[4, 474]" = torch.ops.aten.eq.Scalar(ne_scalar, False);  ne_scalar = None
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4, 474]" = torch.ops.aten.where.self(eq_scalar, full_default, index_tensor);  eq_scalar = full_default = index_tensor = None
        sum_dim_int_list_1: "i64[4]" = torch.ops.aten.sum.dim_IntList(where_self, [-1])
        sum_dim_int_list_2: "i64[4]" = torch.ops.aten.sum.dim_IntList(where_self, [-1]);  where_self = None
        max_default: "i64[]" = torch.ops.aten.max.default(sum_dim_int_list_2);  sum_dim_int_list_2 = None
        return (sum_dim_int_list, sum_dim_int_list_1, max_default)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [4, 474], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [2225], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
