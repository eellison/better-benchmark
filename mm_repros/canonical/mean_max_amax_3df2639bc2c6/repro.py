"""
Standalone repro captured via capture_hook.
Label: gpt-oss-20b
Pattern hash: 3df2639bc2c6
Shape hash: 25c36d41
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
    def forward(self, arg0_1: "bf16[201088, 2880]", arg1_1: "i64[1, 1000]", arg2_1: "bf16[2880]", arg3_1: "bf16[4096, 2880]", arg4_1: "bf16[4096]", arg5_1: "bf16[512, 2880]", arg6_1: "bf16[512]", arg7_1: "bf16[32]", arg8_1: "bf16[64]", arg9_1: "bf16[512, 2880]", arg10_1: "bf16[512]", arg11_1: "bf16[2880, 4096]", arg12_1: "bf16[2880]", arg13_1: "bf16[2880]", arg14_1: "bf16[32, 2880]", arg15_1: "bf16[32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30):
        # No stacktrace found for following nodes
        embedding: "bf16[1, 1000, 2880]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 199999);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_1: "f32[1, 1000, 2880]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        mul_1: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(arg2_1, mul);  arg2_1 = mul = None
        convert_element_type_1: "bf16[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        view: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  _shape_param_0 = None
        permute: "bf16[2880, 4096]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm: "bf16[1000, 4096]" = torch.ops.aten.addmm.default(arg4_1, view, permute);  arg4_1 = view = permute = None
        view_1: "bf16[1, 1000, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        view_2: "bf16[1, 1000, 64, 64]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_1: "bf16[1, 64, 1000, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        split = torch.ops.aten.split.Tensor(permute_1, 32, -1);  permute_1 = None
        getitem: "bf16[1, 64, 1000, 32]" = split[0]
        getitem_1: "bf16[1, 64, 1000, 32]" = split[1];  split = None
        view_3: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  _shape_param_3 = None
        permute_2: "bf16[2880, 512]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg6_1, view_3, permute_2);  arg6_1 = view_3 = permute_2 = None
        view_4: "bf16[1, 1000, 512]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        view_5: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_3: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        split_1 = torch.ops.aten.split.Tensor(permute_3, 32, -1);  permute_3 = None
        getitem_2: "bf16[1, 8, 1000, 32]" = split_1[0]
        getitem_3: "bf16[1, 8, 1000, 32]" = split_1[1];  split_1 = None
        unsqueeze: "bf16[1, 32]" = torch.ops.aten.unsqueeze.default(arg7_1, 0);  arg7_1 = None
        unsqueeze_1: "bf16[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        convert_element_type_2: "f32[1, 32, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        expand: "f32[1, 32, 1]" = torch.ops.aten.expand.default(convert_element_type_2, [1, -1, 1]);  convert_element_type_2 = None
        expand_1: "f32[1, 32, 1]" = torch.ops.aten.expand.default(expand, _shape_param_6);  expand = _shape_param_6 = None
        iota: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1000]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_2: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_3: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        convert_element_type_3: "f32[1, 1, 1000]" = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float32);  unsqueeze_3 = None
        expand_2: "f32[1, 1, 1000]" = torch.ops.aten.expand.default(convert_element_type_3, _shape_param_7);  convert_element_type_3 = _shape_param_7 = None
        mul_2: "f32[1, 32, 1000]" = torch.ops.aten.mul.Tensor(expand_1, expand_2);  expand_1 = expand_2 = None
        permute_4: "f32[1, 1000, 32]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1]);  mul_2 = None
        cos: "f32[1, 1000, 32]" = torch.ops.aten.cos.default(permute_4)
        mul_3: "f32[1, 1000, 32]" = torch.ops.aten.mul.Tensor(cos, 1.3465735902799727);  cos = None
        convert_element_type_4: "bf16[1, 1000, 32]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        unsqueeze_4: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 1)
        mul_4: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_4)
        sin: "f32[1, 1000, 32]" = torch.ops.aten.sin.default(permute_4);  permute_4 = None
        mul_5: "f32[1, 1000, 32]" = torch.ops.aten.mul.Tensor(sin, 1.3465735902799727);  sin = None
        convert_element_type_5: "bf16[1, 1000, 32]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        unsqueeze_5: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1)
        mul_6: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_5)
        sub: "bf16[1, 64, 1000, 32]" = torch.ops.aten.sub.Tensor(mul_4, mul_6);  mul_4 = mul_6 = None
        mul_7: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_4);  getitem_1 = None
        mul_8: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_5);  getitem = None
        add_2: "bf16[1, 64, 1000, 32]" = torch.ops.aten.add.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None
        cat: "bf16[1, 64, 1000, 64]" = torch.ops.aten.cat.default([sub, add_2], -1);  sub = add_2 = None
        expand_3: "bf16[1, 64, 1000, 64]" = torch.ops.aten.expand.default(cat, _shape_param_8);  cat = _shape_param_8 = None
        view_6: "bf16[64, 1000, 64]" = torch.ops.aten.view.default(expand_3, _shape_param_9);  expand_3 = _shape_param_9 = None
        mul_9: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_4)
        mul_10: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_5)
        sub_1: "bf16[1, 8, 1000, 32]" = torch.ops.aten.sub.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        mul_11: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_4);  getitem_3 = unsqueeze_4 = None
        mul_12: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_5);  getitem_2 = unsqueeze_5 = None
        add_3: "bf16[1, 8, 1000, 32]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        cat_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.cat.default([sub_1, add_3], -1);  sub_1 = add_3 = None
        unsqueeze_6: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(cat_1, 2)
        expand_4: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_6, _shape_param_10);  unsqueeze_6 = _shape_param_10 = None
        clone: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_7: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(clone, _shape_param_11);  clone = _shape_param_11 = None
        permute_5: "bf16[1, 64, 64, 1000]" = torch.ops.aten.permute.default(view_7, [0, 1, 3, 2]);  view_7 = None
        expand_5: "bf16[1, 64, 64, 1000]" = torch.ops.aten.expand.default(permute_5, _shape_param_12);  permute_5 = _shape_param_12 = None
        view_8: "bf16[64, 64, 1000]" = torch.ops.aten.view.default(expand_5, _shape_param_13);  expand_5 = _shape_param_13 = None
        bmm: "bf16[64, 1000, 1000]" = torch.ops.aten.bmm.default(view_6, view_8);  view_6 = view_8 = None
        view_9: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.view.default(bmm, _shape_param_14);  bmm = _shape_param_14 = None
        mul_13: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.mul.Tensor(view_9, 0.125);  view_9 = None
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_1: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[1000]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_7: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_8: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        iota_2: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_5: "i64[1000]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None
        unsqueeze_10: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add_5, 0);  add_5 = None
        unsqueeze_11: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
        unsqueeze_12: "i64[1, 1, 1000, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        sub_2: "i64[1, 1, 1000, 1]" = torch.ops.aten.sub.Tensor(unsqueeze_12, 128)
        gt: "b8[1, 1, 1000, 1000]" = torch.ops.aten.gt.Tensor(unsqueeze_9, sub_2);  sub_2 = None
        bitwise_and: "b8[1, 1, 1000, 1000]" = torch.ops.aten.bitwise_and.Tensor(full, gt);  full = gt = None
        le: "b8[1, 1, 1000, 1000]" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_12);  unsqueeze_9 = unsqueeze_12 = None
        bitwise_and_1: "b8[1, 1, 1000, 1000]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, le);  bitwise_and = le = None
        expand_6: "b8[1, 1, 1000, 1000]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_15);  bitwise_and_1 = _shape_param_15 = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_2: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand_6, full_1, full_2);  expand_6 = full_1 = full_2 = None
        add_6: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.add.Tensor(mul_13, where);  mul_13 = None
        view_10: "bf16[1, 64, 1, 1]" = torch.ops.aten.view.default(arg8_1, [1, -1, 1, 1]);  arg8_1 = None
        expand_7: "bf16[1, 64, 1000, 1]" = torch.ops.aten.expand.default(view_10, _shape_param_16);  view_10 = _shape_param_16 = None
        cat_2: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.cat.default([add_6, expand_7], -1);  add_6 = expand_7 = None
        max_1 = torch.ops.aten.max.dim(cat_2, -1, True)
        getitem_4: "bf16[1, 64, 1000, 1]" = max_1[0]
        getitem_5: "i64[1, 64, 1000, 1]" = max_1[1];  max_1 = getitem_5 = None
        sub_3: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.sub.Tensor(cat_2, getitem_4);  cat_2 = getitem_4 = None
        convert_element_type_6: "f32[1, 64, 1000, 1001]" = torch.ops.prims.convert_element_type.default(sub_3, torch.float32);  sub_3 = None
        amax: "f32[1, 64, 1000, 1]" = torch.ops.aten.amax.default(convert_element_type_6, [-1], True)
        sub_4: "f32[1, 64, 1000, 1001]" = torch.ops.aten.sub.Tensor(convert_element_type_6, amax);  convert_element_type_6 = amax = None
        exp: "f32[1, 64, 1000, 1001]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_1: "f32[1, 64, 1000, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 64, 1000, 1001]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_7: "bf16[1, 64, 1000, 1001]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        slice_1: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.slice.Tensor(convert_element_type_7, 3, 0, -1);  convert_element_type_7 = None
        clone_1: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.clone.default(slice_1);  slice_1 = None
        expand_8: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.expand.default(clone_1, _shape_param_17);  clone_1 = _shape_param_17 = None
        view_11: "bf16[64, 1000, 1000]" = torch.ops.aten.view.default(expand_8, _shape_param_18);  expand_8 = _shape_param_18 = None
        view_12: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_19);  convert_element_type_1 = _shape_param_19 = None
        permute_6: "bf16[2880, 512]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg10_1, view_12, permute_6);  arg10_1 = view_12 = permute_6 = None
        view_13: "bf16[1, 1000, 512]" = torch.ops.aten.view.default(addmm_2, _shape_param_20);  addmm_2 = _shape_param_20 = None
        view_14: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(view_13, _shape_param_21);  view_13 = _shape_param_21 = None
        permute_7: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        unsqueeze_13: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(permute_7, 2)
        expand_9: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_13, _shape_param_22);  unsqueeze_13 = _shape_param_22 = None
        clone_2: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_15: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_23);  clone_2 = _shape_param_23 = None
        expand_10: "bf16[1, 64, 1000, 64]" = torch.ops.aten.expand.default(view_15, _shape_param_24);  view_15 = _shape_param_24 = None
        view_16: "bf16[64, 1000, 64]" = torch.ops.aten.view.default(expand_10, _shape_param_25);  expand_10 = _shape_param_25 = None
        bmm_1: "bf16[64, 1000, 64]" = torch.ops.aten.bmm.default(view_11, view_16);  view_11 = view_16 = None
        view_17: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_26);  bmm_1 = _shape_param_26 = None
        permute_8: "bf16[1, 1000, 64, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_3: "bf16[1, 1000, 64, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_18: "bf16[1, 1000, 4096]" = torch.ops.aten.view.default(clone_3, _shape_param_27);  clone_3 = _shape_param_27 = None
        view_19: "bf16[1000, 4096]" = torch.ops.aten.view.default(view_18, _shape_param_28);  view_18 = _shape_param_28 = None
        permute_9: "bf16[4096, 2880]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[1000, 2880]" = torch.ops.aten.addmm.default(arg12_1, view_19, permute_9);  arg12_1 = view_19 = permute_9 = None
        view_20: "bf16[1, 1000, 2880]" = torch.ops.aten.view.default(addmm_3, _shape_param_29);  addmm_3 = _shape_param_29 = None
        add_7: "bf16[1, 1000, 2880]" = torch.ops.aten.add.Tensor(embedding, view_20);  embedding = view_20 = None
        convert_element_type_8: "f32[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        pow_2: "f32[1, 1000, 2880]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 2)
        mean_1: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_8: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_1: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_14: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_8, rsqrt_1);  convert_element_type_8 = rsqrt_1 = None
        mul_15: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(arg13_1, mul_14);  arg13_1 = mul_14 = None
        convert_element_type_9: "bf16[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        view_21: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_30);  convert_element_type_9 = _shape_param_30 = None
        permute_10: "bf16[2880, 32]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_4: "bf16[1000, 32]" = torch.ops.aten.addmm.default(arg15_1, view_21, permute_10);  arg15_1 = permute_10 = None
        topk = torch.ops.aten.topk.default(addmm_4, 4);  addmm_4 = None
        getitem_6: "bf16[1000, 4]" = topk[0]
        getitem_7: "i64[1000, 4]" = topk[1];  topk = None
        view_22: "i64[4000]" = torch.ops.aten.view.default(getitem_7, [-1]);  getitem_7 = None
        sort = torch.ops.aten.sort.default(view_22);  view_22 = None
        getitem_8: "i64[4000]" = sort[0]
        getitem_9: "i64[4000]" = sort[1];  sort = None
        convert_element_type_10: "i32[4000]" = torch.ops.prims.convert_element_type.default(getitem_8, torch.int32)
        slice_2: "bf16[1, 8, 127, 64]" = torch.ops.aten.slice.Tensor(permute_7, 2, -127, 9223372036854775807);  permute_7 = None
        slice_3: "bf16[1, 8, 127, 64]" = torch.ops.aten.slice.Tensor(cat_1, 2, -127, 9223372036854775807);  cat_1 = None
        return (convert_element_type_4, convert_element_type_5, where, add_7, view_21, getitem_6, getitem_8, getitem_9, convert_element_type_10, slice_2, slice_3)



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
