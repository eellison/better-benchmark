"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-4-5-linux.aws.a100_graph10
Pattern hash: 4a937216dd36
Shape hash: db913cf7
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([50265, 1024], bf16), T([1, 256], i64, gen=Index(50265)), T([514, 1024], bf16), T([1024], bf16), T([1024], bf16), T([1024, 1024], bf16), T([1024, 1024], bf16), S([256, 1024]), S([256, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "bf16[50265, 1024]", arg0_1: "i64[1, 256]", arg2_1: "bf16[514, 1024]", arg3_1: "bf16[1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[1024, 1024]", arg7_1: "bf16[1024, 1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding_default: "bf16[1, 256, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "bf16[1, 256, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None
        iota_default: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        expand_default: "i64[1, 256]" = torch.ops.aten.expand.default(iota_default, [1, -1]);  iota_default = None
        add_tensor: "i64[1, 256]" = torch.ops.aten.add.Tensor(expand_default, 2);  expand_default = None
        embedding_default_1: "bf16[1, 256, 1024]" = torch.ops.aten.embedding.default(arg2_1, add_tensor);  arg2_1 = add_tensor = None
        add_tensor_1: "bf16[1, 256, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None
        convert_element_type_default: "f32[1, 256, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[1, 256, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor_1: "f32[1, 256, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[1, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg3_1);  mul_tensor_1 = arg3_1 = None
        add_tensor_3: "f32[1, 256, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg4_1);  mul_tensor_2 = arg4_1 = None
        convert_element_type_default_1: "bf16[1, 256, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        view_default: "bf16[256, 1024]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_0);  _shape_param_0 = None
        permute_default: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        view_default_1: "bf16[256, 1024]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        permute_default_1: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        return (view_default, permute_default, view_default_1, permute_default_1)


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
