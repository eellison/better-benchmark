"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_001
Pattern hash: 38c955b52137
Shape hash: 7ed52c3f
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
_shapes_config = "(T([4, 16, 512, 128], bf16), S([4, 8, 2, 512, 128]), S([4, 512, 1024]), S([2048, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_83: "bf16[4, 16, 512, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.view.default(getitem_83, _shape_param_0);  getitem_83 = _shape_param_0 = None
        sum_dim_int_list: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_default, [2], True);  view_default = None
        squeeze_dim: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 2);  sum_dim_int_list = None
        permute_default: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_dim, [0, 2, 1, 3]);  squeeze_dim = None
        clone_default: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        view_default_2: "bf16[2048, 1024]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default_1: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
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
