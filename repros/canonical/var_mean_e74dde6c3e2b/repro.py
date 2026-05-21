"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_infer_000
Pattern hash: e74dde6c3e2b
Shape hash: 1d3b25e9
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
_shapes_config = "(T([8008, 2560], f32), T([32, 128], i64, gen=Index(8008)), T([128, 2560], f32), T([2560], f32), T([2560], f32), S([4096, 2560]), S([4096, 2560]), S([4096, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[8008, 2560]", arg0_1: "i64[32, 128]", arg2_1: "f32[128, 2560]", arg3_1: "f32[2560]", arg4_1: "f32[2560]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding_default: "f32[32, 128, 2560]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        embedding_default_1: "f32[128, 2560]" = torch.ops.aten.embedding.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None
        add_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg3_1);  mul_tensor_1 = arg3_1 = None
        add_tensor_3: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg4_1);  mul_tensor_2 = arg4_1 = None
        view_default: "f32[4096, 2560]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_0);  _shape_param_0 = None
        view_default_1: "f32[4096, 2560]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[4096, 2560]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
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
