"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer_000
Pattern hash: 68e4555a2a35
Shape hash: d279edd3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128100, 1536], f32), T([8, 512], i64, gen=Index(128100)), T([512, 1536], f32), T([1, 512], i64, gen=Index(512)), T([1536], f32), T([1536], f32), S([4096, 1536]), S([4096, 1536]), S([4096, 1536]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[128100, 1536]", arg0_1: "i64[8, 512]", arg3_1: "f32[512, 1536]", arg1_1: "i64[1, 512]", arg4_1: "f32[1536]", arg5_1: "f32[1536]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding_default: "f32[8, 512, 1536]" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg2_1 = arg0_1 = None
        embedding_default_1: "f32[1, 512, 1536]" = torch.ops.aten.embedding.default(arg3_1, arg1_1);  arg3_1 = arg1_1 = None
        add_tensor: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor, arg4_1);  mul_tensor = arg4_1 = None
        add_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg5_1);  mul_tensor_1 = arg5_1 = None
        view_default: "f32[4096, 1536]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_0);  _shape_param_0 = None
        view_default_1: "f32[4096, 1536]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[4096, 1536]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        return (view_default, view_default_1, view_default_2)

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
