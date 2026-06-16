"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train
Pattern hash: ef83fb4a9dbc
Shape hash: 9a73f5a0
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
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "f32[32000, 768]", arg2_1: "i64[32, 512]", arg3_1: "f32[4, 768]", arg4_1: "f32[512, 768]", arg5_1: "i64[1, 512]", arg6_1: "f32[768]", arg7_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg2_1, 3);  arg1_1 = arg2_1 = None
        embedding_1: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, expand);  arg3_1 = expand = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  arg6_1 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        view: "f32[16384, 768]" = torch.ops.aten.view.default(add_3, _shape_param_1);  add_3 = _shape_param_1 = None
        div: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (mul, view, div)



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
