"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Roberta_base_infer_000
Pattern hash: a4b0facb2977
Shape hash: 16484795
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([250002, 768], f16), T([1, 512], i64, gen=Index(250002)), T([1, 514], i64, gen=Index(1)), T([1, 512], i64, gen=Index(513)), T([1, 512], i32, gen=Index(2)), T([1, 768], f16), T([514, 768], f16), T([768], f16), T([768], f16), S([1, 512]), S([512, 768]), S([512, 768]), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f16[250002, 768]", arg0_1: "i64[1, 512]", arg1_1: "i64[1, 514]", cumsum: "i64[1, 512]", convert_element_type: "i32[1, 512]", arg3_1: "f16[1, 768]", arg4_1: "f16[514, 768]", arg5_1: "f16[768]", arg6_1: "f16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        embedding_default: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 1);  arg2_1 = arg0_1 = None
        expand_default: "i64[1, 514]" = torch.ops.aten.expand.default(arg1_1, [1, -1]);  arg1_1 = None
        convert_element_type_default: "i32[1, 512]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add_tensor: "i32[1, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default, 0);  convert_element_type_default = None
        mul_tensor: "i32[1, 512]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type);  add_tensor = convert_element_type = None
        convert_element_type_default_1: "i64[1, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor_1: "i64[1, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        gather_default: "i64[1, 512]" = torch.ops.aten.gather.default(expand_default, 1, add_tensor_1);  expand_default = None
        expand_default_1: "i64[1, 512]" = torch.ops.aten.expand.default(gather_default, _shape_param_0);  gather_default = _shape_param_0 = None
        embedding_default_1: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, expand_default_1);  arg3_1 = expand_default_1 = None
        add_tensor_2: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        embedding_default_2: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, add_tensor_1, 1);  arg4_1 = add_tensor_1 = None
        add_tensor_3: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, embedding_default_2);  add_tensor_2 = embedding_default_2 = None
        convert_element_type_default_2: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1);  convert_element_type_default_2 = getitem_1 = None
        add_tensor_4: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg5_1);  mul_tensor_1 = arg5_1 = None
        add_tensor_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg6_1);  mul_tensor_2 = arg6_1 = None
        convert_element_type_default_3: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float16);  add_tensor_5 = None
        view_default: "f16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_1);  _shape_param_1 = None
        view_default_1: "f16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_2);  _shape_param_2 = None
        view_default_2: "f16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_3);  convert_element_type_default_3 = _shape_param_3 = None
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
