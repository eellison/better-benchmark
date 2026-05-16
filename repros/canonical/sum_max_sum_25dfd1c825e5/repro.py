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
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 474]", arg1_1: "i64[2225]"):
        # File: <eval_with_key>.0 from /tmp/pytorch-work/torch/fx/experimental/proxy_tensor.py:1721 in wrapped:5 in forward, code: ne = torch.ops.aten.ne.Scalar(arg0_1, 0)
        ne_scalar: "b8[4, 474]" = torch.ops.aten.ne.Scalar(arg0_1, 0)

        # File: <eval_with_key>.0 from /tmp/pytorch-work/torch/fx/experimental/proxy_tensor.py:1721 in wrapped:8 in forward, code: eq = torch.ops.aten.eq.Scalar(ne, False)
        eq_scalar: "b8[4, 474]" = torch.ops.aten.eq.Scalar(ne_scalar, False)

        # File: <eval_with_key>.0 from /tmp/pytorch-work/torch/fx/experimental/proxy_tensor.py:1721 in wrapped:9 in forward, code: full = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: <eval_with_key>.0 from /tmp/pytorch-work/torch/fx/experimental/proxy_tensor.py:1721 in wrapped:7 in forward, code: index = torch.ops.aten.index.Tensor(arg1_1, [arg0_1]);  arg1_1 = arg0_1 = None
        index_tensor: "i64[4, 474]" = torch.ops.aten.index.Tensor(arg1_1, [arg0_1]);  arg1_1 = arg0_1 = None

        # File: <eval_with_key>.0 from /tmp/pytorch-work/torch/fx/experimental/proxy_tensor.py:1721 in wrapped:10 in forward, code: where = torch.ops.aten.where.self(eq, full, index);  eq = full = index = None
        where_self: "i64[4, 474]" = torch.ops.aten.where.self(eq_scalar, full_default, index_tensor);  eq_scalar = full_default = index_tensor = None

        # File: <eval_with_key>.0 from /tmp/pytorch-work/torch/fx/experimental/proxy_tensor.py:1721 in wrapped:12 in forward, code: sum_3 = torch.ops.aten.sum.dim_IntList(where, [-1])
        sum_dim_int_list: "i64[4]" = torch.ops.aten.sum.dim_IntList(where_self, [-1])

        # File: <eval_with_key>.0 from /tmp/pytorch-work/torch/fx/experimental/proxy_tensor.py:1721 in wrapped:13 in forward, code: max_1 = torch.ops.aten.max.default(sum_3);  sum_3 = None
        max_default: "i64[]" = torch.ops.aten.max.default(sum_dim_int_list);  sum_dim_int_list = None

        # File: <eval_with_key>.0 from /tmp/pytorch-work/torch/fx/experimental/proxy_tensor.py:1721 in wrapped:6 in forward, code: sum_1 = torch.ops.aten.sum.dim_IntList(ne, [-1])
        sum_dim_int_list_1: "i64[4]" = torch.ops.aten.sum.dim_IntList(ne_scalar, [-1]);  ne_scalar = None

        # File: <eval_with_key>.0 from /tmp/pytorch-work/torch/fx/experimental/proxy_tensor.py:1721 in wrapped:11 in forward, code: sum_2 = torch.ops.aten.sum.dim_IntList(where, [-1])
        sum_dim_int_list_2: "i64[4]" = torch.ops.aten.sum.dim_IntList(where_self, [-1]);  where_self = None
        return (max_default, sum_dim_int_list_1, sum_dim_int_list_2)


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
