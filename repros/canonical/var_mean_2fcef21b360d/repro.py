"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_infer_000
Pattern hash: 2fcef21b360d
Shape hash: 19f7d9f6
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
_shapes_config = "(T([32000, 768], f32), T([32, 512], i64, gen=Index(32000)), T([1, 512], i64, gen=Index(4)), T([4, 768], f32), T([512, 768], f32), T([1, 512], i64, gen=Index(512)), T([768], f32), T([768], f32), S([32, 512]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[32000, 768]", arg0_1: "i64[32, 512]", arg1_1: "i64[1, 512]", arg4_1: "f32[4, 768]", arg5_1: "f32[512, 768]", arg2_1: "i64[1, 512]", arg6_1: "f32[768]", arg7_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding_default: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 3);  arg3_1 = arg0_1 = None
        expand_default: "i64[32, 512]" = torch.ops.aten.expand.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        embedding_default_1: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, expand_default);  arg4_1 = expand_default = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        embedding_default_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, arg2_1);  arg5_1 = arg2_1 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg6_1);  mul_tensor = arg6_1 = None
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg7_1);  mul_tensor_1 = arg7_1 = None
        view_default: "f32[16384, 768]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
        return view_default



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
