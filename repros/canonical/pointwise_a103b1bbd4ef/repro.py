"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_000
Pattern hash: a103b1bbd4ef
Shape hash: d7517139
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
_shapes_config = "(S([4, -1]))"

class Repro(torch.nn.Module):
    def forward(self, _shape_param_0):
        # No stacktrace found for following nodes
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        expand_default: "i64[4, 512]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        slice_tensor: "i64[4, 1]" = torch.ops.aten.slice.Tensor(expand_default, 1, 0, 1)
        sub_tensor: "i64[4, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None
        cat_default: "i64[4, 513]" = torch.ops.aten.cat.default([sub_tensor, expand_default], -1);  sub_tensor = expand_default = None
        slice_tensor_1: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 513)
        slice_tensor_2: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 512);  cat_default = None
        sub_tensor_1: "i64[4, 512]" = torch.ops.aten.sub.Tensor(slice_tensor_1, slice_tensor_2);  slice_tensor_1 = slice_tensor_2 = None
        ne_scalar: "b8[4, 512]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None
        return ne_scalar



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
