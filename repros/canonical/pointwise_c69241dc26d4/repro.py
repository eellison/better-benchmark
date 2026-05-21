"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_infer_000
Pattern hash: c69241dc26d4
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
_shapes_config = "()"

class Repro(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        slice_tensor: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze_default, 1, 0, 1)
        sub_tensor: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None
        cat_default: "i64[1, 129]" = torch.ops.aten.cat.default([sub_tensor, unsqueeze_default], -1);  sub_tensor = unsqueeze_default = None
        slice_tensor_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 129)
        slice_tensor_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 128);  cat_default = None
        sub_tensor_1: "i64[1, 128]" = torch.ops.aten.sub.Tensor(slice_tensor_1, slice_tensor_2);  slice_tensor_1 = slice_tensor_2 = None
        ne_scalar: "b8[1, 128]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None
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
