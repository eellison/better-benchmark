"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_train_001
Pattern hash: afd118ca907d
Shape hash: 5e25ca33
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([64, 1024, 1024], f32), T([8, 8, 1024, 1024], b8), T([8, 8, 1024, 1024], f32), T([1024, 1024], i64, gen=Index(32)), T([], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([64, 1024, 1024], f32), T([8, 8, 1024, 1024], b8), T([1, 1, 1024, 1], b8), T([64, 1024, 1024], f32), T([1024, 1024, 8], f32), T([8, 8, 1024, 1], f32), T([8, 8, 1024, 1], f32), T([1024, 1024], i64, gen=Index(32)), S([8, 8, 1024, 1024]), S([64, 1024, 1024]), S([8, 8, 1024, 1024]), S([64, 1024, 1024]), S([8, 8, 1024, 1024]), S([8, -1, 1024, 1024]), S([8, 8, 1024, 1024]), S([64, 1024, 1024]), S([8, 8, 1024, 1024]), S([64, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, view_40: "f32[8, 8, 1024, 1024]", view_87: "f32[8, 8, 1024, 1024]", view_134: "f32[8, 8, 1024, 1024]", view_181: "f32[8, 8, 1024, 1024]", view_228: "f32[8, 8, 1024, 1024]", bmm_45: "f32[64, 1024, 1024]", arg226_1: "b8[8, 8, 1024, 1024]", arg225_1: "f32[8, 8, 1024, 1024]", arg224_1: "i64[1024, 1024]", full_1: "f32[]", view_302: "f32[8, 8, 1024, 1024]", view_328: "f32[8, 8, 1024, 1024]", view_354: "f32[8, 8, 1024, 1024]", view_380: "f32[8, 8, 1024, 1024]", view_406: "f32[8, 8, 1024, 1024]", bmm_69: "f32[64, 1024, 1024]", arg141_1: "b8[8, 8, 1024, 1024]", arg132_1: "b8[1, 1, 1024, 1]", arg136_1: "f32[64, 1024, 1024]", arg138_1: "f32[1024, 1024, 8]", arg139_1: "f32[8, 8, 1024, 1]", arg140_1: "f32[8, 8, 1024, 1]", arg137_1: "i64[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        add_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_40, view_87);  view_40 = view_87 = None
        add_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor, view_134);  add_tensor = view_134 = None
        add_tensor_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, view_181);  add_tensor_1 = view_181 = None
        add_tensor_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, view_228);  add_tensor_2 = view_228 = None
        view_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_45, _shape_param_0);  bmm_45 = _shape_param_0 = None
        convert_element_type_default: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg226_1, torch.float32);  arg226_1 = None
        mul_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = mul_tensor = None
        mul_tensor_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg225_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(arg225_1);  arg225_1 = None
        fma_default: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        view_default_1: "f32[64, 1024, 1024]" = torch.ops.aten.view.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        view_default_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        view_default_3: "f32[64, 1024, 1024]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  _shape_param_3 = None
        add_tensor_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_3, view_default_2);  add_tensor_3 = view_default_2 = None
        sum_dim_int_list_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_4, [0], True);  add_tensor_4 = None
        squeeze_dim: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None
        permute_default: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None
        eq_scalar: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(arg224_1, -1)
        unsqueeze_default: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default, full_1, permute_default);  unsqueeze_default = permute_default = None
        clone_default: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        full_default: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default, [arg224_1], clone_default, True);  arg224_1 = clone_default = None
        add_tensor_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_302, view_328);  view_302 = view_328 = None
        add_tensor_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_5, view_354);  add_tensor_5 = view_354 = None
        add_tensor_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_6, view_380);  add_tensor_6 = view_380 = None
        add_tensor_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_7, view_406);  add_tensor_7 = view_406 = None
        view_default_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_69, _shape_param_4);  bmm_69 = _shape_param_4 = None
        convert_element_type_default_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float32);  arg141_1 = None
        mul_tensor_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_default_4, mul_tensor_3);  view_default_4 = mul_tensor_3 = None
        expand_default: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(arg132_1, _shape_param_5);  arg132_1 = _shape_param_5 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_1, full_default_1);  expand_default = full_default_1 = None
        view_default_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg136_1, _shape_param_6);  arg136_1 = _shape_param_6 = None
        permute_default_1: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(arg138_1, [2, 0, 1]);  arg138_1 = None
        unsqueeze_default_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_default_1, 0);  permute_default_1 = None
        add_tensor_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, where_self_1);  unsqueeze_default_1 = where_self_1 = None
        add_tensor_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_default_5, add_tensor_9);  view_default_5 = add_tensor_9 = None
        sub_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_10, arg139_1);  add_tensor_10 = arg139_1 = None
        exp_default: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, arg140_1);  exp_default = arg140_1 = None
        mul_tensor_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_4, div_tensor);  mul_tensor_4 = None
        sum_dim_int_list_2: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [-1], True)
        neg_default_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_default_1, sum_dim_int_list_2, mul_tensor_5);  neg_default_1 = sum_dim_int_list_2 = mul_tensor_5 = None
        view_default_6: "f32[64, 1024, 1024]" = torch.ops.aten.view.default(fma_default_1, _shape_param_7);  fma_default_1 = _shape_param_7 = None
        view_default_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_default_6, _shape_param_8);  view_default_6 = _shape_param_8 = None
        view_default_8: "f32[64, 1024, 1024]" = torch.ops.aten.view.default(view_default_7, _shape_param_9);  _shape_param_9 = None
        add_tensor_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_8, view_default_7);  add_tensor_8 = view_default_7 = None
        sum_dim_int_list_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_11, [0], True);  add_tensor_11 = None
        squeeze_dim_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_3, 0);  sum_dim_int_list_3 = None
        permute_default_2: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None
        eq_scalar_1: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(arg137_1, -1)
        unsqueeze_default_2: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_2: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default_2, full_1, permute_default_2);  unsqueeze_default_2 = full_1 = permute_default_2 = None
        clone_default_1: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self_2, memory_format = torch.contiguous_format);  where_self_2 = None
        index_put_default_1: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default, [arg137_1], clone_default_1, True);  full_default = arg137_1 = clone_default_1 = None
        return (view_default_3, index_put_default, view_default_8, index_put_default_1)

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
