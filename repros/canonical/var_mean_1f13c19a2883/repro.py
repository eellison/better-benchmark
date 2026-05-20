"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-3-5-linux.aws.a100_graph7
Pattern hash: 1f13c19a2883
Shape hash: 0e1c83ea
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([30522, 768], f32), T([1, 512], i64, max=30522), T([512, 768], f32), T([1, 512], i64, max=512), T([1024, 768], f32), T([1024, 768], f32), T([1024, 768], f32), T([1024, 768], f32), T([2, 768], f32), T([768], f32), T([768], f32), T([768], f32), T([768, 768], f32), T([768], f32), T([768, 768], f32), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[30522, 768]", arg0_1: "i64[1, 512]", arg3_1: "f32[512, 768]", arg2_1: "i64[1, 512]", arg4_1: "f32[1024, 768]", arg5_1: "f32[1024, 768]", arg6_1: "f32[1024, 768]", arg7_1: "f32[1024, 768]", arg8_1: "f32[2, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg12_1: "f32[768]", arg11_1: "f32[768, 768]", arg14_1: "f32[768]", arg13_1: "f32[768, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        full_default: "i64[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "i64[1, 512, 4]" = torch.ops.aten.full.default([1, 512, 4], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        embedding_default: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        embedding_default_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg2_1);  arg3_1 = arg2_1 = None
        select_int: "i64[1, 512]" = torch.ops.aten.select.int(full_default_1, 2, 0)
        embedding_default_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select_int)
        select_int_1: "i64[1, 512]" = torch.ops.aten.select.int(full_default_1, 2, 1)
        embedding_default_3: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_int_1)
        select_int_2: "i64[1, 512]" = torch.ops.aten.select.int(full_default_1, 2, 2)
        embedding_default_4: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select_int_2);  arg4_1 = None
        select_int_3: "i64[1, 512]" = torch.ops.aten.select.int(full_default_1, 2, 3);  full_default_1 = None
        embedding_default_5: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_int_3);  arg5_1 = None
        sub_tensor: "i64[1, 512]" = torch.ops.aten.sub.Tensor(select_int_3, select_int_1);  select_int_3 = select_int_1 = None
        embedding_default_6: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg6_1, sub_tensor);  arg6_1 = sub_tensor = None
        sub_tensor_1: "i64[1, 512]" = torch.ops.aten.sub.Tensor(select_int_2, select_int);  select_int_2 = select_int = None
        embedding_default_7: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg7_1, sub_tensor_1);  arg7_1 = sub_tensor_1 = None
        embedding_default_8: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg8_1, full_default);  arg8_1 = full_default = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        add_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default_3);  add_tensor_1 = embedding_default_3 = None
        add_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, embedding_default_4);  add_tensor_2 = embedding_default_4 = None
        add_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, embedding_default_5);  add_tensor_3 = embedding_default_5 = None
        add_tensor_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, embedding_default_6);  add_tensor_4 = embedding_default_6 = None
        add_tensor_6: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_5, embedding_default_7);  add_tensor_5 = embedding_default_7 = None
        add_tensor_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_6, embedding_default_8);  add_tensor_6 = embedding_default_8 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_7, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        sub_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_7, getitem_1);  add_tensor_7 = getitem_1 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg9_1);  mul_tensor = arg9_1 = None
        add_tensor_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg10_1);  mul_tensor_1 = arg10_1 = None
        native_dropout_default = torch.ops.aten.native_dropout.default(add_tensor_9, 0.1, True);  add_tensor_9 = None
        getitem_2: "f32[1, 512, 768]" = native_dropout_default[0]
        getitem_3: "b8[1, 512, 768]" = native_dropout_default[1];  native_dropout_default = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float16);  arg12_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float16);  arg11_1 = None
        convert_element_type_default_2: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float16);  getitem_2 = None
        view_default: "f16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_2, _shape_param_0);  convert_element_type_default_2 = _shape_param_0 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        convert_element_type_default_3: "f16[768]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float16);  arg14_1 = None
        convert_element_type_default_4: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float16);  arg13_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        return (convert_element_type_default, view_default, permute_default, convert_element_type_default_3, permute_default_1, getitem_3)


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
