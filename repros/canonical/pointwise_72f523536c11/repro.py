"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train_000
Pattern hash: 72f523536c11
Shape hash: e6af15e5
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
_shapes_config = "(T([1, 128], i64), S([32, -1]))"

class Repro(torch.nn.Module):
    def forward(self, unsqueeze: "i64[1, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        expand_default: "i64[32, 128]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        slice_tensor: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand_default, 1, 0, 1)
        sub_tensor: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None
        cat_default: "i64[32, 129]" = torch.ops.aten.cat.default([sub_tensor, expand_default], -1);  sub_tensor = expand_default = None
        slice_tensor_1: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 128)
        slice_tensor_2: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 129);  cat_default = None
        sub_tensor_1: "i64[32, 128]" = torch.ops.aten.sub.Tensor(slice_tensor_2, slice_tensor_1);  slice_tensor_2 = slice_tensor_1 = None
        ne_scalar: "b8[32, 128]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None
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
