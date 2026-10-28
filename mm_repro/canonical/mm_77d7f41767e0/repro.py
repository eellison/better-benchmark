"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/GPTNeoForSequenceClassification/full_graph_000.py
Pattern hash: 77d7f41767e0
Shape hash: 08a60a4a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[2, 2048]", arg1_1: "bf16[4096, 2048]", arg2_1: "i64[32, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        permute: "bf16[2048, 2]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[4096, 2]" = torch.ops.aten.mm.default(arg1_1, permute);  arg1_1 = permute = None
        view: "bf16[32, 128, 2]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_1: "i32[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        ne: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg2_1, 0);  arg2_1 = None
        convert_element_type: "i32[32, 128]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
        mul: "i32[32, 128]" = torch.ops.aten.mul.Tensor(iota_1, convert_element_type);  iota_1 = convert_element_type = None
        argmax: "i64[32]" = torch.ops.aten.argmax.default(mul, -1);  mul = None
        index: "bf16[32, 2]" = torch.ops.aten.index.Tensor(view, [iota, argmax]);  view = iota = argmax = None
        convert_element_type_1: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(index, torch.float32)
        return (index, convert_element_type_1)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
