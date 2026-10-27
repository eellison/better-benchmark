"""
Standalone repro captured via capture_hook.
Label: DistilBertForMaskedLM
Pattern hash: 5219c127f368
Shape hash: c117bf2c
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[256, 128]", arg3_1: "bf16[256, 128, 30522]", arg4_1: "f32[32768, 1]", arg5_1: "f32[32768, 1]", arg6_1: "bf16[256, 128, 30522]", arg7_1: "bf16[30522, 768]", arg8_1: "bf16[32768, 768]", arg9_1: "f32[768]", arg10_1: "bf16[32768, 768]", arg11_1: "f32[256, 128, 1]", arg12_1: "f32[256, 128, 1]", arg13_1: "bf16[768, 768]", arg14_1: "bf16[32768, 768]", arg15_1: "f32[768]", arg16_1: "f32[256, 128, 768]", arg17_1: "f32[256, 128, 1]", arg18_1: "b8[256, 128, 768]", arg19_1: "bf16[768, 3072]", arg20_1: "bf16[32768, 3072]", arg21_1: "bf16[32768, 3072]", arg22_1: "bf16[3072, 768]", arg23_1: "bf16[32768, 768]", arg24_1: "f32[768]", arg25_1: "f32[256, 128, 768]", arg26_1: "f32[256, 128, 1]", arg27_1: "bf16[768, 768]", arg28_1: "bf16[32768, 768]", arg29_1: "bf16[3072, 128, 128]", arg30_1: "bf16[3072, 64, 128]", arg31_1: "b8[256, 12, 128, 128]", arg32_1: "bf16[256, 12, 128, 128]", arg33_1: "bf16[3072, 64, 128]", arg34_1: "bf16[3072, 128, 64]", arg35_1: "bf16[768, 768]", arg36_1: "bf16[32768, 768]", arg37_1: "bf16[768, 768]", arg38_1: "bf16[768, 768]", arg39_1: "f32[768]", arg40_1: "f32[256, 128, 768]", arg41_1: "f32[256, 128, 1]", arg42_1: "b8[256, 128, 768]", arg43_1: "bf16[768, 3072]", arg44_1: "bf16[32768, 3072]", arg45_1: "bf16[32768, 3072]", arg46_1: "bf16[3072, 768]", arg47_1: "bf16[32768, 768]", arg48_1: "f32[768]", arg49_1: "f32[256, 128, 768]", arg50_1: "f32[256, 128, 1]", arg51_1: "bf16[768, 768]", arg52_1: "bf16[32768, 768]", arg53_1: "bf16[3072, 128, 128]", arg54_1: "bf16[3072, 64, 128]", arg55_1: "b8[256, 12, 128, 128]", arg56_1: "bf16[256, 12, 128, 128]", arg57_1: "bf16[3072, 64, 128]", arg58_1: "bf16[3072, 128, 64]", arg59_1: "bf16[768, 768]", arg60_1: "bf16[32768, 768]", arg61_1: "bf16[768, 768]", arg62_1: "bf16[768, 768]", arg63_1: "f32[768]", arg64_1: "f32[256, 128, 768]", arg65_1: "f32[256, 128, 1]", arg66_1: "b8[256, 128, 768]", arg67_1: "bf16[768, 3072]", arg68_1: "bf16[32768, 3072]", arg69_1: "bf16[32768, 3072]", arg70_1: "bf16[3072, 768]", arg71_1: "bf16[32768, 768]", arg72_1: "f32[768]", arg73_1: "f32[256, 128, 768]", arg74_1: "f32[256, 128, 1]", arg75_1: "bf16[768, 768]", arg76_1: "bf16[32768, 768]", arg77_1: "bf16[3072, 128, 128]", arg78_1: "bf16[3072, 64, 128]", arg79_1: "b8[256, 12, 128, 128]", arg80_1: "bf16[256, 12, 128, 128]", arg81_1: "bf16[3072, 64, 128]", arg82_1: "bf16[3072, 128, 64]", arg83_1: "bf16[768, 768]", arg84_1: "bf16[32768, 768]", arg85_1: "bf16[768, 768]", arg86_1: "bf16[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[256, 128, 768]", arg89_1: "f32[256, 128, 1]", arg90_1: "b8[256, 128, 768]", arg91_1: "bf16[768, 3072]", arg92_1: "bf16[32768, 3072]", arg93_1: "bf16[32768, 3072]", arg94_1: "bf16[3072, 768]", arg95_1: "bf16[32768, 768]", arg96_1: "f32[768]", arg97_1: "f32[256, 128, 768]", arg98_1: "f32[256, 128, 1]", arg99_1: "bf16[768, 768]", arg100_1: "bf16[32768, 768]", arg101_1: "bf16[3072, 128, 128]", arg102_1: "bf16[3072, 64, 128]", arg103_1: "b8[256, 12, 128, 128]", arg104_1: "bf16[256, 12, 128, 128]", arg105_1: "bf16[3072, 64, 128]", arg106_1: "bf16[3072, 128, 64]", arg107_1: "bf16[768, 768]", arg108_1: "bf16[32768, 768]", arg109_1: "bf16[768, 768]", arg110_1: "bf16[768, 768]", arg111_1: "f32[768]", arg112_1: "f32[256, 128, 768]", arg113_1: "f32[256, 128, 1]", arg114_1: "b8[256, 128, 768]", arg115_1: "bf16[768, 3072]", arg116_1: "bf16[32768, 3072]", arg117_1: "bf16[32768, 3072]", arg118_1: "bf16[3072, 768]", arg119_1: "bf16[32768, 768]", arg120_1: "f32[768]", arg121_1: "f32[256, 128, 768]", arg122_1: "f32[256, 128, 1]", arg123_1: "bf16[768, 768]", arg124_1: "bf16[32768, 768]", arg125_1: "bf16[3072, 128, 128]", arg126_1: "bf16[3072, 64, 128]", arg127_1: "b8[256, 12, 128, 128]", arg128_1: "bf16[256, 12, 128, 128]", arg129_1: "bf16[3072, 64, 128]", arg130_1: "bf16[3072, 128, 64]", arg131_1: "bf16[768, 768]", arg132_1: "bf16[32768, 768]", arg133_1: "bf16[768, 768]", arg134_1: "bf16[768, 768]", arg135_1: "f32[768]", arg136_1: "f32[256, 128, 768]", arg137_1: "f32[256, 128, 1]", arg138_1: "b8[256, 128, 768]", arg139_1: "bf16[768, 3072]", arg140_1: "bf16[32768, 3072]", arg141_1: "bf16[32768, 3072]", arg142_1: "bf16[3072, 768]", arg143_1: "bf16[32768, 768]", arg144_1: "f32[768]", arg145_1: "f32[256, 128, 768]", arg146_1: "f32[256, 128, 1]", arg147_1: "bf16[768, 768]", arg148_1: "bf16[32768, 768]", arg149_1: "bf16[3072, 128, 128]", arg150_1: "bf16[3072, 64, 128]", arg151_1: "b8[256, 12, 128, 128]", arg152_1: "bf16[3072, 128, 128]", arg153_1: "f32[256, 12, 128, 1]", arg154_1: "f32[256, 12, 128, 1]", arg155_1: "b8[256, 12, 128, 1]", arg156_1: "bf16[3072, 64, 128]", arg157_1: "bf16[3072, 128, 64]", arg158_1: "bf16[768, 768]", arg159_1: "bf16[32768, 768]", arg160_1: "bf16[768, 768]", arg161_1: "bf16[768, 768]", arg162_1: "b8[256, 128, 768]", arg163_1: "f32[768]", arg164_1: "f32[256, 128, 768]", arg165_1: "f32[1, 128, 768]", arg166_1: "f32[256, 128, 1]", arg167_1: "f32[256, 128, 1]", arg168_1: "i64[1, 512]", arg169_1: "i64[256, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[32768]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[32768, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[32768, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 30522]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[32768, 30522]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[32768, 30522]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[32768, 30522]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[32768, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[32768, 30522]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[32768, 30522]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[32768, 30522]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[256, 128, 30522]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[32768, 30522]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        constant_pad_nd: "bf16[32768, 30528]" = torch.ops.aten.constant_pad_nd.default(view_4, _shape_param_5);  _shape_param_5 = None
        constant_pad_nd_1: "bf16[30528, 768]" = torch.ops.aten.constant_pad_nd.default(arg7_1, _shape_param_6);  arg7_1 = _shape_param_6 = None
        mm: "bf16[32768, 768]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        permute: "bf16[30522, 32768]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_1: "bf16[30522, 768]" = torch.ops.aten.mm.default(permute, arg8_1);  permute = arg8_1 = None
        sum_2: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[30522]" = torch.ops.aten.view.default(sum_2, _shape_param_7);  sum_2 = _shape_param_7 = None
        convert_element_type_6: "bf16[30522]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm, _shape_param_8);  mm = _shape_param_8 = None
        convert_element_type_7: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_8: "f32[30522, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_9: "f32[30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        mul_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, arg9_1);  arg9_1 = None
        mul_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_3: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        view_7: "bf16[256, 128, 768]" = torch.ops.aten.view.default(arg10_1, _shape_param_9);  arg10_1 = _shape_param_9 = None
        convert_element_type_10: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        mul_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.5)
        mul_5: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.7071067811865476)
        erf: "f32[256, 128, 768]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_1: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_4, add_1);  mul_4 = None
        convert_element_type_11: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_12: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        sub_3: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_12, arg11_1);  convert_element_type_12 = arg11_1 = None
        mul_7: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, arg12_1);  sub_3 = None
        mul_8: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, mul_7);  mul_2 = None
        sum_4: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True);  mul_8 = None
        mul_9: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_4);  sum_4 = None
        sub_4: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_5: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_4, mul_9);  sub_4 = mul_9 = None
        div_1: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(arg12_1, 768);  arg12_1 = None
        mul_10: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_11: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, mul_7);  mul_7 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 1]);  mul_11 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0, 1]);  convert_element_type_7 = None
        convert_element_type_13: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        convert_element_type_14: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        mul_12: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_13: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, convert_element_type_10)
        mul_14: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_13, -0.5);  mul_13 = None
        exp_1: "f32[256, 128, 768]" = torch.ops.aten.exp.default(mul_14);  mul_14 = None
        mul_15: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_16: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_15);  convert_element_type_10 = mul_15 = None
        add_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_12, mul_16);  mul_12 = mul_16 = None
        mul_17: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, add_2);  convert_element_type_14 = add_2 = None
        convert_element_type_15: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_8: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_10);  convert_element_type_15 = _shape_param_10 = None
        mm_2: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_8, arg13_1);  arg13_1 = None
        permute_1: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_3: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_1, arg14_1);  permute_1 = arg14_1 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_11);  sum_7 = _shape_param_11 = None
        convert_element_type_16: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_12);  mm_2 = _shape_param_12 = None
        convert_element_type_17: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_18: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_19: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        mul_18: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, arg15_1);  arg15_1 = None
        mul_19: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, 768)
        sum_8: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
        mul_20: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg16_1);  mul_18 = None
        sum_9: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
        mul_21: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg16_1, sum_9);  sum_9 = None
        sub_6: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_19, sum_8);  mul_19 = sum_8 = None
        sub_7: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_6, mul_21);  sub_6 = mul_21 = None
        mul_22: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg17_1, sub_7);  arg17_1 = sub_7 = None
        mul_23: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, arg16_1);  arg16_1 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_17, [0, 1]);  convert_element_type_17 = None
        convert_element_type_20: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16)
        convert_element_type_21: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        mul_24: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_25: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, mul_24);  convert_element_type_20 = mul_24 = None
        view_11: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_25, _shape_param_13);  mul_25 = _shape_param_13 = None
        mm_4: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_11, arg19_1);  arg19_1 = None
        permute_2: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_5: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_2, arg20_1);  permute_2 = arg20_1 = None
        sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[768]" = torch.ops.aten.view.default(sum_12, _shape_param_14);  sum_12 = _shape_param_14 = None
        convert_element_type_22: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(mm_4, _shape_param_15);  mm_4 = _shape_param_15 = None
        convert_element_type_23: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_24: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        convert_element_type_25: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        view_14: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(arg21_1, _shape_param_16);  arg21_1 = _shape_param_16 = None
        convert_element_type_26: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        mul_26: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476)
        erf_1: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_3: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_27: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        mul_28: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_26, convert_element_type_26)
        mul_29: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_28, -0.5);  mul_28 = None
        exp_2: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_29);  mul_29 = None
        mul_30: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_31: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_26, mul_30);  convert_element_type_26 = mul_30 = None
        add_4: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_27, mul_31);  mul_27 = mul_31 = None
        mul_32: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_25, add_4);  convert_element_type_25 = add_4 = None
        convert_element_type_27: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        view_15: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_17);  convert_element_type_27 = _shape_param_17 = None
        mm_6: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_15, arg22_1);  arg22_1 = None
        permute_3: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_7: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_3, arg23_1);  permute_3 = arg23_1 = None
        sum_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True, dtype = torch.float32);  view_15 = None
        view_16: "f32[3072]" = torch.ops.aten.view.default(sum_13, _shape_param_18);  sum_13 = _shape_param_18 = None
        convert_element_type_28: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        view_17: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_19);  mm_6 = _shape_param_19 = None
        convert_element_type_29: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        add_5: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_22, convert_element_type_29);  mul_22 = convert_element_type_29 = None
        convert_element_type_30: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_31: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32);  convert_element_type_28 = None
        mul_33: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_5, arg24_1);  arg24_1 = None
        mul_34: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_33, 768)
        sum_14: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_33, [2], True)
        mul_35: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg25_1);  mul_33 = None
        sum_15: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True);  mul_35 = None
        mul_36: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg25_1, sum_15);  sum_15 = None
        sub_8: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_34, sum_14);  mul_34 = sum_14 = None
        sub_9: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_8, mul_36);  sub_8 = mul_36 = None
        mul_37: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg26_1, sub_9);  arg26_1 = sub_9 = None
        mul_38: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_5, arg25_1);  arg25_1 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 1]);  mul_38 = None
        sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_5, [0, 1]);  add_5 = None
        convert_element_type_32: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16)
        view_18: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_32, _shape_param_20);  convert_element_type_32 = _shape_param_20 = None
        mm_8: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_18, arg27_1);  arg27_1 = None
        permute_4: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_9: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_4, arg28_1);  permute_4 = arg28_1 = None
        sum_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True, dtype = torch.float32);  view_18 = None
        view_19: "f32[768]" = torch.ops.aten.view.default(sum_18, _shape_param_21);  sum_18 = _shape_param_21 = None
        convert_element_type_33: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_8, _shape_param_22);  mm_8 = _shape_param_22 = None
        convert_element_type_34: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_35: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_33, torch.float32);  convert_element_type_33 = None
        view_21: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_20, _shape_param_23);  view_20 = _shape_param_23 = None
        permute_5: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_22: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_24);  clone = _shape_param_24 = None
        bmm: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(arg29_1, view_22);  arg29_1 = None
        bmm_1: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_22, arg30_1);  view_22 = arg30_1 = None
        view_23: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm, _shape_param_25);  bmm = _shape_param_25 = None
        view_24: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_26);  bmm_1 = _shape_param_26 = None
        convert_element_type_36: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        mul_39: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_40: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_24, mul_39);  view_24 = mul_39 = None
        convert_element_type_37: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_40, torch.float32);  mul_40 = None
        convert_element_type_38: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.float32);  arg32_1 = None
        mul_41: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_37, convert_element_type_38);  convert_element_type_37 = None
        sum_19: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_41, [-1], True)
        neg: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_38);  convert_element_type_38 = None
        fma: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg, sum_19, mul_41);  neg = sum_19 = mul_41 = None
        convert_element_type_39: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_25: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(convert_element_type_39, _shape_param_27);  convert_element_type_39 = _shape_param_27 = None
        bmm_2: "bf16[3072, 64, 128]" = torch.ops.aten.bmm.default(arg33_1, view_25);  arg33_1 = None
        bmm_3: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_25, arg34_1);  view_25 = arg34_1 = None
        view_26: "bf16[256, 12, 64, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_28);  bmm_2 = _shape_param_28 = None
        view_27: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_29);  bmm_3 = _shape_param_29 = None
        mul_42: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_26, 0.3535533905932738);  view_26 = None
        permute_6: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_42, [0, 1, 3, 2]);  mul_42 = None
        mul_43: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_27, 0.3535533905932738);  view_27 = None
        permute_7: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        clone_1: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_28: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_1, _shape_param_30);  clone_1 = _shape_param_30 = None
        view_29: "bf16[32768, 768]" = torch.ops.aten.view.default(view_28, _shape_param_31);  view_28 = _shape_param_31 = None
        mm_10: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_29, arg35_1);  arg35_1 = None
        permute_8: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_29, [1, 0])
        mm_11: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_8, arg36_1);  permute_8 = None
        sum_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_29, [0], True, dtype = torch.float32);  view_29 = None
        view_30: "f32[768]" = torch.ops.aten.view.default(sum_20, _shape_param_32);  sum_20 = _shape_param_32 = None
        convert_element_type_40: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_30, torch.bfloat16);  view_30 = None
        view_31: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_10, _shape_param_33);  mm_10 = _shape_param_33 = None
        convert_element_type_41: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        add_6: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_37, convert_element_type_41);  mul_37 = convert_element_type_41 = None
        convert_element_type_42: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_43: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_40, torch.float32);  convert_element_type_40 = None
        permute_9: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1, 3]);  permute_6 = None
        view_32: "bf16[256, 128, 768]" = torch.ops.aten.view.default(permute_9, _shape_param_34);  permute_9 = _shape_param_34 = None
        clone_2: "bf16[256, 128, 768]" = torch.ops.aten.clone.default(view_32, memory_format = torch.contiguous_format);  view_32 = None
        view_33: "bf16[32768, 768]" = torch.ops.aten.view.default(clone_2, _shape_param_35);  clone_2 = _shape_param_35 = None
        mm_12: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_33, arg37_1);  arg37_1 = None
        permute_10: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_33, [1, 0])
        mm_13: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_10, arg36_1);  permute_10 = None
        sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_33, [0], True, dtype = torch.float32);  view_33 = None
        view_34: "f32[768]" = torch.ops.aten.view.default(sum_21, _shape_param_36);  sum_21 = _shape_param_36 = None
        convert_element_type_44: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_34, torch.bfloat16);  view_34 = None
        view_35: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_12, _shape_param_37);  mm_12 = _shape_param_37 = None
        convert_element_type_45: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        add_7: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_45);  add_6 = convert_element_type_45 = None
        convert_element_type_46: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_47: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_44, torch.float32);  convert_element_type_44 = None
        permute_11: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_43, [0, 2, 1, 3]);  mul_43 = None
        clone_3: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_36: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_3, _shape_param_38);  clone_3 = _shape_param_38 = None
        view_37: "bf16[32768, 768]" = torch.ops.aten.view.default(view_36, _shape_param_39);  view_36 = _shape_param_39 = None
        mm_14: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_37, arg38_1);  arg38_1 = None
        permute_12: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_15: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_12, arg36_1);  permute_12 = arg36_1 = None
        sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True, dtype = torch.float32);  view_37 = None
        view_38: "f32[768]" = torch.ops.aten.view.default(sum_22, _shape_param_40);  sum_22 = _shape_param_40 = None
        convert_element_type_48: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        view_39: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_14, _shape_param_41);  mm_14 = _shape_param_41 = None
        convert_element_type_49: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        add_8: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_49);  add_7 = convert_element_type_49 = None
        convert_element_type_50: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_51: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_48, torch.float32);  convert_element_type_48 = None
        mul_44: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_8, arg39_1);  arg39_1 = None
        mul_45: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_44, 768)
        sum_23: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_44, [2], True)
        mul_46: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg40_1);  mul_44 = None
        sum_24: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_46, [2], True);  mul_46 = None
        mul_47: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg40_1, sum_24);  sum_24 = None
        sub_10: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_45, sum_23);  mul_45 = sum_23 = None
        sub_11: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_10, mul_47);  sub_10 = mul_47 = None
        mul_48: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg41_1, sub_11);  arg41_1 = sub_11 = None
        mul_49: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_8, arg40_1);  arg40_1 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_49, [0, 1]);  mul_49 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_8, [0, 1]);  add_8 = None
        convert_element_type_52: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16)
        convert_element_type_53: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        mul_50: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
        mul_51: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_52, mul_50);  convert_element_type_52 = mul_50 = None
        view_40: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_51, _shape_param_42);  mul_51 = _shape_param_42 = None
        mm_16: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_40, arg43_1);  arg43_1 = None
        permute_13: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_17: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_13, arg44_1);  permute_13 = arg44_1 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True, dtype = torch.float32);  view_40 = None
        view_41: "f32[768]" = torch.ops.aten.view.default(sum_27, _shape_param_43);  sum_27 = _shape_param_43 = None
        convert_element_type_54: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_41, torch.bfloat16);  view_41 = None
        view_42: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(mm_16, _shape_param_44);  mm_16 = _shape_param_44 = None
        convert_element_type_55: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_56: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_54, torch.float32);  convert_element_type_54 = None
        convert_element_type_57: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_42, torch.float32);  view_42 = None
        view_43: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(arg45_1, _shape_param_45);  arg45_1 = _shape_param_45 = None
        convert_element_type_58: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        mul_52: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.7071067811865476)
        erf_2: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_9: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_53: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_54: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_58, convert_element_type_58)
        mul_55: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_54, -0.5);  mul_54 = None
        exp_3: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_55);  mul_55 = None
        mul_56: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_57: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_58, mul_56);  convert_element_type_58 = mul_56 = None
        add_10: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_53, mul_57);  mul_53 = mul_57 = None
        mul_58: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_57, add_10);  convert_element_type_57 = add_10 = None
        convert_element_type_59: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None
        view_44: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_59, _shape_param_46);  convert_element_type_59 = _shape_param_46 = None
        mm_18: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_44, arg46_1);  arg46_1 = None
        permute_14: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_44, [1, 0])
        mm_19: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_14, arg47_1);  permute_14 = arg47_1 = None
        sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True, dtype = torch.float32);  view_44 = None
        view_45: "f32[3072]" = torch.ops.aten.view.default(sum_28, _shape_param_47);  sum_28 = _shape_param_47 = None
        convert_element_type_60: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_45, torch.bfloat16);  view_45 = None
        view_46: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_18, _shape_param_48);  mm_18 = _shape_param_48 = None
        convert_element_type_61: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        add_11: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_48, convert_element_type_61);  mul_48 = convert_element_type_61 = None
        convert_element_type_62: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_63: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_60, torch.float32);  convert_element_type_60 = None
        mul_59: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_11, arg48_1);  arg48_1 = None
        mul_60: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_59, 768)
        sum_29: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_59, [2], True)
        mul_61: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_59, arg49_1);  mul_59 = None
        sum_30: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_61, [2], True);  mul_61 = None
        mul_62: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg49_1, sum_30);  sum_30 = None
        sub_12: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_60, sum_29);  mul_60 = sum_29 = None
        sub_13: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_12, mul_62);  sub_12 = mul_62 = None
        mul_63: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg50_1, sub_13);  arg50_1 = sub_13 = None
        mul_64: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_11, arg49_1);  arg49_1 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_64, [0, 1]);  mul_64 = None
        sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
        convert_element_type_64: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16)
        view_47: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_64, _shape_param_49);  convert_element_type_64 = _shape_param_49 = None
        mm_20: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_47, arg51_1);  arg51_1 = None
        permute_15: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_21: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_15, arg52_1);  permute_15 = arg52_1 = None
        sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True, dtype = torch.float32);  view_47 = None
        view_48: "f32[768]" = torch.ops.aten.view.default(sum_33, _shape_param_50);  sum_33 = _shape_param_50 = None
        convert_element_type_65: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_48, torch.bfloat16);  view_48 = None
        view_49: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_20, _shape_param_51);  mm_20 = _shape_param_51 = None
        convert_element_type_66: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_67: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_65, torch.float32);  convert_element_type_65 = None
        view_50: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_49, _shape_param_52);  view_49 = _shape_param_52 = None
        permute_16: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_4: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_51: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_53);  clone_4 = _shape_param_53 = None
        bmm_4: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(arg53_1, view_51);  arg53_1 = None
        bmm_5: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_51, arg54_1);  view_51 = arg54_1 = None
        view_52: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_4, _shape_param_54);  bmm_4 = _shape_param_54 = None
        view_53: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_5, _shape_param_55);  bmm_5 = _shape_param_55 = None
        convert_element_type_68: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        mul_65: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_66: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_53, mul_65);  view_53 = mul_65 = None
        convert_element_type_69: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_66, torch.float32);  mul_66 = None
        convert_element_type_70: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.float32);  arg56_1 = None
        mul_67: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_69, convert_element_type_70);  convert_element_type_69 = None
        sum_34: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_67, [-1], True)
        neg_1: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_70);  convert_element_type_70 = None
        fma_1: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_1, sum_34, mul_67);  neg_1 = sum_34 = mul_67 = None
        convert_element_type_71: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        view_54: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_56);  convert_element_type_71 = _shape_param_56 = None
        bmm_6: "bf16[3072, 64, 128]" = torch.ops.aten.bmm.default(arg57_1, view_54);  arg57_1 = None
        bmm_7: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_54, arg58_1);  view_54 = arg58_1 = None
        view_55: "bf16[256, 12, 64, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_57);  bmm_6 = _shape_param_57 = None
        view_56: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_58);  bmm_7 = _shape_param_58 = None
        mul_68: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_55, 0.3535533905932738);  view_55 = None
        permute_17: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_68, [0, 1, 3, 2]);  mul_68 = None
        mul_69: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_56, 0.3535533905932738);  view_56 = None
        permute_18: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_5: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_57: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_5, _shape_param_59);  clone_5 = _shape_param_59 = None
        view_58: "bf16[32768, 768]" = torch.ops.aten.view.default(view_57, _shape_param_60);  view_57 = _shape_param_60 = None
        mm_22: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_58, arg59_1);  arg59_1 = None
        permute_19: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_58, [1, 0])
        mm_23: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_19, arg60_1);  permute_19 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_58, [0], True, dtype = torch.float32);  view_58 = None
        view_59: "f32[768]" = torch.ops.aten.view.default(sum_35, _shape_param_61);  sum_35 = _shape_param_61 = None
        convert_element_type_72: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_59, torch.bfloat16);  view_59 = None
        view_60: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_22, _shape_param_62);  mm_22 = _shape_param_62 = None
        convert_element_type_73: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_60, torch.float32);  view_60 = None
        add_12: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_63, convert_element_type_73);  mul_63 = convert_element_type_73 = None
        convert_element_type_74: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_75: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_72, torch.float32);  convert_element_type_72 = None
        permute_20: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_17, [0, 2, 1, 3]);  permute_17 = None
        view_61: "bf16[256, 128, 768]" = torch.ops.aten.view.default(permute_20, _shape_param_63);  permute_20 = _shape_param_63 = None
        clone_6: "bf16[256, 128, 768]" = torch.ops.aten.clone.default(view_61, memory_format = torch.contiguous_format);  view_61 = None
        view_62: "bf16[32768, 768]" = torch.ops.aten.view.default(clone_6, _shape_param_64);  clone_6 = _shape_param_64 = None
        mm_24: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_62, arg61_1);  arg61_1 = None
        permute_21: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_62, [1, 0])
        mm_25: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_21, arg60_1);  permute_21 = None
        sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_62, [0], True, dtype = torch.float32);  view_62 = None
        view_63: "f32[768]" = torch.ops.aten.view.default(sum_36, _shape_param_65);  sum_36 = _shape_param_65 = None
        convert_element_type_76: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_63, torch.bfloat16);  view_63 = None
        view_64: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_24, _shape_param_66);  mm_24 = _shape_param_66 = None
        convert_element_type_77: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_64, torch.float32);  view_64 = None
        add_13: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_12, convert_element_type_77);  add_12 = convert_element_type_77 = None
        convert_element_type_78: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_79: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_76, torch.float32);  convert_element_type_76 = None
        permute_22: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_69, [0, 2, 1, 3]);  mul_69 = None
        clone_7: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_65: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_7, _shape_param_67);  clone_7 = _shape_param_67 = None
        view_66: "bf16[32768, 768]" = torch.ops.aten.view.default(view_65, _shape_param_68);  view_65 = _shape_param_68 = None
        mm_26: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_66, arg62_1);  arg62_1 = None
        permute_23: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_66, [1, 0])
        mm_27: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_23, arg60_1);  permute_23 = arg60_1 = None
        sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_66, [0], True, dtype = torch.float32);  view_66 = None
        view_67: "f32[768]" = torch.ops.aten.view.default(sum_37, _shape_param_69);  sum_37 = _shape_param_69 = None
        convert_element_type_80: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_67, torch.bfloat16);  view_67 = None
        view_68: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_26, _shape_param_70);  mm_26 = _shape_param_70 = None
        convert_element_type_81: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        add_14: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_81);  add_13 = convert_element_type_81 = None
        convert_element_type_82: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_83: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_80, torch.float32);  convert_element_type_80 = None
        mul_70: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_14, arg63_1);  arg63_1 = None
        mul_71: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_70, 768)
        sum_38: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_70, [2], True)
        mul_72: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg64_1);  mul_70 = None
        sum_39: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_72, [2], True);  mul_72 = None
        mul_73: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg64_1, sum_39);  sum_39 = None
        sub_14: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_71, sum_38);  mul_71 = sum_38 = None
        sub_15: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_14, mul_73);  sub_14 = mul_73 = None
        mul_74: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg65_1, sub_15);  arg65_1 = sub_15 = None
        mul_75: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_14, arg64_1);  arg64_1 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_75, [0, 1]);  mul_75 = None
        sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
        convert_element_type_84: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_74, torch.bfloat16)
        convert_element_type_85: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(arg66_1, torch.bfloat16);  arg66_1 = None
        mul_76: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 1.1111111111111112);  convert_element_type_85 = None
        mul_77: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_84, mul_76);  convert_element_type_84 = mul_76 = None
        view_69: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_77, _shape_param_71);  mul_77 = _shape_param_71 = None
        mm_28: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_69, arg67_1);  arg67_1 = None
        permute_24: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_69, [1, 0])
        mm_29: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_24, arg68_1);  permute_24 = arg68_1 = None
        sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_69, [0], True, dtype = torch.float32);  view_69 = None
        view_70: "f32[768]" = torch.ops.aten.view.default(sum_42, _shape_param_72);  sum_42 = _shape_param_72 = None
        convert_element_type_86: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_70, torch.bfloat16);  view_70 = None
        view_71: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(mm_28, _shape_param_73);  mm_28 = _shape_param_73 = None
        convert_element_type_87: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_88: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_86, torch.float32);  convert_element_type_86 = None
        convert_element_type_89: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_71, torch.float32);  view_71 = None
        view_72: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(arg69_1, _shape_param_74);  arg69_1 = _shape_param_74 = None
        convert_element_type_90: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_72, torch.float32);  view_72 = None
        mul_78: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 0.7071067811865476)
        erf_3: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_15: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_79: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.5);  add_15 = None
        mul_80: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_90, convert_element_type_90)
        mul_81: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_80, -0.5);  mul_80 = None
        exp_4: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_81);  mul_81 = None
        mul_82: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_83: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_90, mul_82);  convert_element_type_90 = mul_82 = None
        add_16: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_79, mul_83);  mul_79 = mul_83 = None
        mul_84: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_89, add_16);  convert_element_type_89 = add_16 = None
        convert_element_type_91: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None
        view_73: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_91, _shape_param_75);  convert_element_type_91 = _shape_param_75 = None
        mm_30: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_73, arg70_1);  arg70_1 = None
        permute_25: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_73, [1, 0])
        mm_31: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_25, arg71_1);  permute_25 = arg71_1 = None
        sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_73, [0], True, dtype = torch.float32);  view_73 = None
        view_74: "f32[3072]" = torch.ops.aten.view.default(sum_43, _shape_param_76);  sum_43 = _shape_param_76 = None
        convert_element_type_92: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_74, torch.bfloat16);  view_74 = None
        view_75: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_30, _shape_param_77);  mm_30 = _shape_param_77 = None
        convert_element_type_93: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        add_17: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_74, convert_element_type_93);  mul_74 = convert_element_type_93 = None
        convert_element_type_94: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_95: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_92, torch.float32);  convert_element_type_92 = None
        mul_85: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_17, arg72_1);  arg72_1 = None
        mul_86: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_85, 768)
        sum_44: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_85, [2], True)
        mul_87: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg73_1);  mul_85 = None
        sum_45: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [2], True);  mul_87 = None
        mul_88: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg73_1, sum_45);  sum_45 = None
        sub_16: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_86, sum_44);  mul_86 = sum_44 = None
        sub_17: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_16, mul_88);  sub_16 = mul_88 = None
        mul_89: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg74_1, sub_17);  arg74_1 = sub_17 = None
        mul_90: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_17, arg73_1);  arg73_1 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_90, [0, 1]);  mul_90 = None
        sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_17, [0, 1]);  add_17 = None
        convert_element_type_96: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16)
        view_76: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_96, _shape_param_78);  convert_element_type_96 = _shape_param_78 = None
        mm_32: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_76, arg75_1);  arg75_1 = None
        permute_26: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_76, [1, 0])
        mm_33: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_26, arg76_1);  permute_26 = arg76_1 = None
        sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_76, [0], True, dtype = torch.float32);  view_76 = None
        view_77: "f32[768]" = torch.ops.aten.view.default(sum_48, _shape_param_79);  sum_48 = _shape_param_79 = None
        convert_element_type_97: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_77, torch.bfloat16);  view_77 = None
        view_78: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_32, _shape_param_80);  mm_32 = _shape_param_80 = None
        convert_element_type_98: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_99: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_97, torch.float32);  convert_element_type_97 = None
        view_79: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_78, _shape_param_81);  view_78 = _shape_param_81 = None
        permute_27: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        clone_8: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_80: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_82);  clone_8 = _shape_param_82 = None
        bmm_8: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(arg77_1, view_80);  arg77_1 = None
        bmm_9: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_80, arg78_1);  view_80 = arg78_1 = None
        view_81: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_83);  bmm_8 = _shape_param_83 = None
        view_82: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_9, _shape_param_84);  bmm_9 = _shape_param_84 = None
        convert_element_type_100: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg79_1, torch.bfloat16);  arg79_1 = None
        mul_91: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 1.1111111111111112);  convert_element_type_100 = None
        mul_92: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_82, mul_91);  view_82 = mul_91 = None
        convert_element_type_101: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_92, torch.float32);  mul_92 = None
        convert_element_type_102: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.float32);  arg80_1 = None
        mul_93: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_101, convert_element_type_102);  convert_element_type_101 = None
        sum_49: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_93, [-1], True)
        neg_2: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_102);  convert_element_type_102 = None
        fma_2: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_49, mul_93);  neg_2 = sum_49 = mul_93 = None
        convert_element_type_103: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        view_83: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(convert_element_type_103, _shape_param_85);  convert_element_type_103 = _shape_param_85 = None
        bmm_10: "bf16[3072, 64, 128]" = torch.ops.aten.bmm.default(arg81_1, view_83);  arg81_1 = None
        bmm_11: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_83, arg82_1);  view_83 = arg82_1 = None
        view_84: "bf16[256, 12, 64, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_86);  bmm_10 = _shape_param_86 = None
        view_85: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_87);  bmm_11 = _shape_param_87 = None
        mul_94: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_84, 0.3535533905932738);  view_84 = None
        permute_28: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_94, [0, 1, 3, 2]);  mul_94 = None
        mul_95: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_85, 0.3535533905932738);  view_85 = None
        permute_29: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        clone_9: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_86: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_9, _shape_param_88);  clone_9 = _shape_param_88 = None
        view_87: "bf16[32768, 768]" = torch.ops.aten.view.default(view_86, _shape_param_89);  view_86 = _shape_param_89 = None
        mm_34: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_87, arg83_1);  arg83_1 = None
        permute_30: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_87, [1, 0])
        mm_35: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_30, arg84_1);  permute_30 = None
        sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_87, [0], True, dtype = torch.float32);  view_87 = None
        view_88: "f32[768]" = torch.ops.aten.view.default(sum_50, _shape_param_90);  sum_50 = _shape_param_90 = None
        convert_element_type_104: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_88, torch.bfloat16);  view_88 = None
        view_89: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_34, _shape_param_91);  mm_34 = _shape_param_91 = None
        convert_element_type_105: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_89, torch.float32);  view_89 = None
        add_18: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_89, convert_element_type_105);  mul_89 = convert_element_type_105 = None
        convert_element_type_106: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_107: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_104, torch.float32);  convert_element_type_104 = None
        permute_31: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_90: "bf16[256, 128, 768]" = torch.ops.aten.view.default(permute_31, _shape_param_92);  permute_31 = _shape_param_92 = None
        clone_10: "bf16[256, 128, 768]" = torch.ops.aten.clone.default(view_90, memory_format = torch.contiguous_format);  view_90 = None
        view_91: "bf16[32768, 768]" = torch.ops.aten.view.default(clone_10, _shape_param_93);  clone_10 = _shape_param_93 = None
        mm_36: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_91, arg85_1);  arg85_1 = None
        permute_32: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_91, [1, 0])
        mm_37: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_32, arg84_1);  permute_32 = None
        sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_91, [0], True, dtype = torch.float32);  view_91 = None
        view_92: "f32[768]" = torch.ops.aten.view.default(sum_51, _shape_param_94);  sum_51 = _shape_param_94 = None
        convert_element_type_108: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_92, torch.bfloat16);  view_92 = None
        view_93: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_36, _shape_param_95);  mm_36 = _shape_param_95 = None
        convert_element_type_109: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        add_19: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_18, convert_element_type_109);  add_18 = convert_element_type_109 = None
        convert_element_type_110: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_111: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_108, torch.float32);  convert_element_type_108 = None
        permute_33: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_95, [0, 2, 1, 3]);  mul_95 = None
        clone_11: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_94: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_11, _shape_param_96);  clone_11 = _shape_param_96 = None
        view_95: "bf16[32768, 768]" = torch.ops.aten.view.default(view_94, _shape_param_97);  view_94 = _shape_param_97 = None
        mm_38: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_95, arg86_1);  arg86_1 = None
        permute_34: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_95, [1, 0])
        mm_39: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_34, arg84_1);  permute_34 = arg84_1 = None
        sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True, dtype = torch.float32);  view_95 = None
        view_96: "f32[768]" = torch.ops.aten.view.default(sum_52, _shape_param_98);  sum_52 = _shape_param_98 = None
        convert_element_type_112: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_96, torch.bfloat16);  view_96 = None
        view_97: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_38, _shape_param_99);  mm_38 = _shape_param_99 = None
        convert_element_type_113: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        add_20: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_19, convert_element_type_113);  add_19 = convert_element_type_113 = None
        convert_element_type_114: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_115: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_112, torch.float32);  convert_element_type_112 = None
        mul_96: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_20, arg87_1);  arg87_1 = None
        mul_97: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_96, 768)
        sum_53: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True)
        mul_98: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg88_1);  mul_96 = None
        sum_54: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_98, [2], True);  mul_98 = None
        mul_99: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg88_1, sum_54);  sum_54 = None
        sub_18: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_97, sum_53);  mul_97 = sum_53 = None
        sub_19: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_18, mul_99);  sub_18 = mul_99 = None
        mul_100: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg89_1, sub_19);  arg89_1 = sub_19 = None
        mul_101: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_20, arg88_1);  arg88_1 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_101, [0, 1]);  mul_101 = None
        sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_20, [0, 1]);  add_20 = None
        convert_element_type_116: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_100, torch.bfloat16)
        convert_element_type_117: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(arg90_1, torch.bfloat16);  arg90_1 = None
        mul_102: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_117, 1.1111111111111112);  convert_element_type_117 = None
        mul_103: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_116, mul_102);  convert_element_type_116 = mul_102 = None
        view_98: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_103, _shape_param_100);  mul_103 = _shape_param_100 = None
        mm_40: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_98, arg91_1);  arg91_1 = None
        permute_35: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_98, [1, 0])
        mm_41: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_35, arg92_1);  permute_35 = arg92_1 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True, dtype = torch.float32);  view_98 = None
        view_99: "f32[768]" = torch.ops.aten.view.default(sum_57, _shape_param_101);  sum_57 = _shape_param_101 = None
        convert_element_type_118: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_99, torch.bfloat16);  view_99 = None
        view_100: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(mm_40, _shape_param_102);  mm_40 = _shape_param_102 = None
        convert_element_type_119: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_120: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_118, torch.float32);  convert_element_type_118 = None
        convert_element_type_121: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_100, torch.float32);  view_100 = None
        view_101: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(arg93_1, _shape_param_103);  arg93_1 = _shape_param_103 = None
        convert_element_type_122: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None
        mul_104: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_122, 0.7071067811865476)
        erf_4: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_21: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_105: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
        mul_106: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_122, convert_element_type_122)
        mul_107: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_106, -0.5);  mul_106 = None
        exp_5: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_107);  mul_107 = None
        mul_108: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_109: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_122, mul_108);  convert_element_type_122 = mul_108 = None
        add_22: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_105, mul_109);  mul_105 = mul_109 = None
        mul_110: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_121, add_22);  convert_element_type_121 = add_22 = None
        convert_element_type_123: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        view_102: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_123, _shape_param_104);  convert_element_type_123 = _shape_param_104 = None
        mm_42: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_102, arg94_1);  arg94_1 = None
        permute_36: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_102, [1, 0])
        mm_43: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_36, arg95_1);  permute_36 = arg95_1 = None
        sum_58: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_102, [0], True, dtype = torch.float32);  view_102 = None
        view_103: "f32[3072]" = torch.ops.aten.view.default(sum_58, _shape_param_105);  sum_58 = _shape_param_105 = None
        convert_element_type_124: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_103, torch.bfloat16);  view_103 = None
        view_104: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_42, _shape_param_106);  mm_42 = _shape_param_106 = None
        convert_element_type_125: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_104, torch.float32);  view_104 = None
        add_23: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_100, convert_element_type_125);  mul_100 = convert_element_type_125 = None
        convert_element_type_126: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_127: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_124, torch.float32);  convert_element_type_124 = None
        mul_111: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_23, arg96_1);  arg96_1 = None
        mul_112: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_111, 768)
        sum_59: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True)
        mul_113: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_111, arg97_1);  mul_111 = None
        sum_60: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_113, [2], True);  mul_113 = None
        mul_114: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg97_1, sum_60);  sum_60 = None
        sub_20: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_112, sum_59);  mul_112 = sum_59 = None
        sub_21: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_20, mul_114);  sub_20 = mul_114 = None
        mul_115: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg98_1, sub_21);  arg98_1 = sub_21 = None
        mul_116: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_23, arg97_1);  arg97_1 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_116, [0, 1]);  mul_116 = None
        sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_23, [0, 1]);  add_23 = None
        convert_element_type_128: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_115, torch.bfloat16)
        view_105: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_128, _shape_param_107);  convert_element_type_128 = _shape_param_107 = None
        mm_44: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_105, arg99_1);  arg99_1 = None
        permute_37: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_105, [1, 0])
        mm_45: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_37, arg100_1);  permute_37 = arg100_1 = None
        sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True, dtype = torch.float32);  view_105 = None
        view_106: "f32[768]" = torch.ops.aten.view.default(sum_63, _shape_param_108);  sum_63 = _shape_param_108 = None
        convert_element_type_129: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_106, torch.bfloat16);  view_106 = None
        view_107: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_44, _shape_param_109);  mm_44 = _shape_param_109 = None
        convert_element_type_130: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_131: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_129, torch.float32);  convert_element_type_129 = None
        view_108: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_107, _shape_param_110);  view_107 = _shape_param_110 = None
        permute_38: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        clone_12: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_109: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_111);  clone_12 = _shape_param_111 = None
        bmm_12: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(arg101_1, view_109);  arg101_1 = None
        bmm_13: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_109, arg102_1);  view_109 = arg102_1 = None
        view_110: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_12, _shape_param_112);  bmm_12 = _shape_param_112 = None
        view_111: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_113);  bmm_13 = _shape_param_113 = None
        convert_element_type_132: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg103_1, torch.bfloat16);  arg103_1 = None
        mul_117: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_132, 1.1111111111111112);  convert_element_type_132 = None
        mul_118: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_111, mul_117);  view_111 = mul_117 = None
        convert_element_type_133: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_118, torch.float32);  mul_118 = None
        convert_element_type_134: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg104_1, torch.float32);  arg104_1 = None
        mul_119: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_133, convert_element_type_134);  convert_element_type_133 = None
        sum_64: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [-1], True)
        neg_3: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_134);  convert_element_type_134 = None
        fma_3: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_64, mul_119);  neg_3 = sum_64 = mul_119 = None
        convert_element_type_135: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        view_112: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(convert_element_type_135, _shape_param_114);  convert_element_type_135 = _shape_param_114 = None
        bmm_14: "bf16[3072, 64, 128]" = torch.ops.aten.bmm.default(arg105_1, view_112);  arg105_1 = None
        bmm_15: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_112, arg106_1);  view_112 = arg106_1 = None
        view_113: "bf16[256, 12, 64, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_115);  bmm_14 = _shape_param_115 = None
        view_114: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_116);  bmm_15 = _shape_param_116 = None
        mul_120: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_113, 0.3535533905932738);  view_113 = None
        permute_39: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_120, [0, 1, 3, 2]);  mul_120 = None
        mul_121: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_114, 0.3535533905932738);  view_114 = None
        permute_40: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        clone_13: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_115: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_13, _shape_param_117);  clone_13 = _shape_param_117 = None
        view_116: "bf16[32768, 768]" = torch.ops.aten.view.default(view_115, _shape_param_118);  view_115 = _shape_param_118 = None
        mm_46: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_116, arg107_1);  arg107_1 = None
        permute_41: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_116, [1, 0])
        mm_47: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_41, arg108_1);  permute_41 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_116, [0], True, dtype = torch.float32);  view_116 = None
        view_117: "f32[768]" = torch.ops.aten.view.default(sum_65, _shape_param_119);  sum_65 = _shape_param_119 = None
        convert_element_type_136: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_117, torch.bfloat16);  view_117 = None
        view_118: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_46, _shape_param_120);  mm_46 = _shape_param_120 = None
        convert_element_type_137: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        add_24: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_115, convert_element_type_137);  mul_115 = convert_element_type_137 = None
        convert_element_type_138: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_139: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_136, torch.float32);  convert_element_type_136 = None
        permute_42: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_39, [0, 2, 1, 3]);  permute_39 = None
        view_119: "bf16[256, 128, 768]" = torch.ops.aten.view.default(permute_42, _shape_param_121);  permute_42 = _shape_param_121 = None
        clone_14: "bf16[256, 128, 768]" = torch.ops.aten.clone.default(view_119, memory_format = torch.contiguous_format);  view_119 = None
        view_120: "bf16[32768, 768]" = torch.ops.aten.view.default(clone_14, _shape_param_122);  clone_14 = _shape_param_122 = None
        mm_48: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_120, arg109_1);  arg109_1 = None
        permute_43: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_120, [1, 0])
        mm_49: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_43, arg108_1);  permute_43 = None
        sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True, dtype = torch.float32);  view_120 = None
        view_121: "f32[768]" = torch.ops.aten.view.default(sum_66, _shape_param_123);  sum_66 = _shape_param_123 = None
        convert_element_type_140: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_121, torch.bfloat16);  view_121 = None
        view_122: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_48, _shape_param_124);  mm_48 = _shape_param_124 = None
        convert_element_type_141: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_122, torch.float32);  view_122 = None
        add_25: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_24, convert_element_type_141);  add_24 = convert_element_type_141 = None
        convert_element_type_142: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_143: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_140, torch.float32);  convert_element_type_140 = None
        permute_44: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_121, [0, 2, 1, 3]);  mul_121 = None
        clone_15: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_123: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_15, _shape_param_125);  clone_15 = _shape_param_125 = None
        view_124: "bf16[32768, 768]" = torch.ops.aten.view.default(view_123, _shape_param_126);  view_123 = _shape_param_126 = None
        mm_50: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_124, arg110_1);  arg110_1 = None
        permute_45: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_124, [1, 0])
        mm_51: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_45, arg108_1);  permute_45 = arg108_1 = None
        sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True, dtype = torch.float32);  view_124 = None
        view_125: "f32[768]" = torch.ops.aten.view.default(sum_67, _shape_param_127);  sum_67 = _shape_param_127 = None
        convert_element_type_144: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_125, torch.bfloat16);  view_125 = None
        view_126: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_50, _shape_param_128);  mm_50 = _shape_param_128 = None
        convert_element_type_145: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_126, torch.float32);  view_126 = None
        add_26: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_25, convert_element_type_145);  add_25 = convert_element_type_145 = None
        convert_element_type_146: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_147: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_144, torch.float32);  convert_element_type_144 = None
        mul_122: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_26, arg111_1);  arg111_1 = None
        mul_123: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
        sum_68: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
        mul_124: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_122, arg112_1);  mul_122 = None
        sum_69: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
        mul_125: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg112_1, sum_69);  sum_69 = None
        sub_22: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_68);  mul_123 = sum_68 = None
        sub_23: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_22, mul_125);  sub_22 = mul_125 = None
        mul_126: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg113_1, sub_23);  arg113_1 = sub_23 = None
        mul_127: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_26, arg112_1);  arg112_1 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
        sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_26, [0, 1]);  add_26 = None
        convert_element_type_148: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16)
        convert_element_type_149: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.bfloat16);  arg114_1 = None
        mul_128: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_149, 1.1111111111111112);  convert_element_type_149 = None
        mul_129: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_148, mul_128);  convert_element_type_148 = mul_128 = None
        view_127: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_129, _shape_param_129);  mul_129 = _shape_param_129 = None
        mm_52: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_127, arg115_1);  arg115_1 = None
        permute_46: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_127, [1, 0])
        mm_53: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_46, arg116_1);  permute_46 = arg116_1 = None
        sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_127, [0], True, dtype = torch.float32);  view_127 = None
        view_128: "f32[768]" = torch.ops.aten.view.default(sum_72, _shape_param_130);  sum_72 = _shape_param_130 = None
        convert_element_type_150: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_128, torch.bfloat16);  view_128 = None
        view_129: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(mm_52, _shape_param_131);  mm_52 = _shape_param_131 = None
        convert_element_type_151: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_152: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_150, torch.float32);  convert_element_type_150 = None
        convert_element_type_153: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        view_130: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(arg117_1, _shape_param_132);  arg117_1 = _shape_param_132 = None
        convert_element_type_154: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        mul_130: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_154, 0.7071067811865476)
        erf_5: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
        add_27: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_131: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_132: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_154, convert_element_type_154)
        mul_133: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_132, -0.5);  mul_132 = None
        exp_6: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_133);  mul_133 = None
        mul_134: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_135: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_154, mul_134);  convert_element_type_154 = mul_134 = None
        add_28: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_131, mul_135);  mul_131 = mul_135 = None
        mul_136: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_153, add_28);  convert_element_type_153 = add_28 = None
        convert_element_type_155: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None
        view_131: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_155, _shape_param_133);  convert_element_type_155 = _shape_param_133 = None
        mm_54: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_131, arg118_1);  arg118_1 = None
        permute_47: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_131, [1, 0])
        mm_55: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_47, arg119_1);  permute_47 = arg119_1 = None
        sum_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True, dtype = torch.float32);  view_131 = None
        view_132: "f32[3072]" = torch.ops.aten.view.default(sum_73, _shape_param_134);  sum_73 = _shape_param_134 = None
        convert_element_type_156: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_132, torch.bfloat16);  view_132 = None
        view_133: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_54, _shape_param_135);  mm_54 = _shape_param_135 = None
        convert_element_type_157: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        add_29: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_126, convert_element_type_157);  mul_126 = convert_element_type_157 = None
        convert_element_type_158: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_159: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_156, torch.float32);  convert_element_type_156 = None
        mul_137: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_29, arg120_1);  arg120_1 = None
        mul_138: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_137, 768)
        sum_74: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True)
        mul_139: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_137, arg121_1);  mul_137 = None
        sum_75: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
        mul_140: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg121_1, sum_75);  sum_75 = None
        sub_24: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_138, sum_74);  mul_138 = sum_74 = None
        sub_25: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_24, mul_140);  sub_24 = mul_140 = None
        mul_141: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg122_1, sub_25);  arg122_1 = sub_25 = None
        mul_142: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_29, arg121_1);  arg121_1 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1]);  mul_142 = None
        sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_29, [0, 1]);  add_29 = None
        convert_element_type_160: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16)
        view_134: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_160, _shape_param_136);  convert_element_type_160 = _shape_param_136 = None
        mm_56: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_134, arg123_1);  arg123_1 = None
        permute_48: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_134, [1, 0])
        mm_57: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_48, arg124_1);  permute_48 = arg124_1 = None
        sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_134, [0], True, dtype = torch.float32);  view_134 = None
        view_135: "f32[768]" = torch.ops.aten.view.default(sum_78, _shape_param_137);  sum_78 = _shape_param_137 = None
        convert_element_type_161: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_135, torch.bfloat16);  view_135 = None
        view_136: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_56, _shape_param_138);  mm_56 = _shape_param_138 = None
        convert_element_type_162: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_163: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_161, torch.float32);  convert_element_type_161 = None
        view_137: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_136, _shape_param_139);  view_136 = _shape_param_139 = None
        permute_49: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_16: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_138: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_140);  clone_16 = _shape_param_140 = None
        bmm_16: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(arg125_1, view_138);  arg125_1 = None
        bmm_17: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_138, arg126_1);  view_138 = arg126_1 = None
        view_139: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_16, _shape_param_141);  bmm_16 = _shape_param_141 = None
        view_140: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_17, _shape_param_142);  bmm_17 = _shape_param_142 = None
        convert_element_type_164: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        mul_143: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_164, 1.1111111111111112);  convert_element_type_164 = None
        mul_144: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_140, mul_143);  view_140 = mul_143 = None
        convert_element_type_165: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_144, torch.float32);  mul_144 = None
        convert_element_type_166: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.float32);  arg128_1 = None
        mul_145: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_165, convert_element_type_166);  convert_element_type_165 = None
        sum_79: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [-1], True)
        neg_4: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_166);  convert_element_type_166 = None
        fma_4: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_4, sum_79, mul_145);  neg_4 = sum_79 = mul_145 = None
        convert_element_type_167: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        view_141: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(convert_element_type_167, _shape_param_143);  convert_element_type_167 = _shape_param_143 = None
        bmm_18: "bf16[3072, 64, 128]" = torch.ops.aten.bmm.default(arg129_1, view_141);  arg129_1 = None
        bmm_19: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_141, arg130_1);  view_141 = arg130_1 = None
        view_142: "bf16[256, 12, 64, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_144);  bmm_18 = _shape_param_144 = None
        view_143: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_145);  bmm_19 = _shape_param_145 = None
        mul_146: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_142, 0.3535533905932738);  view_142 = None
        permute_50: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_146, [0, 1, 3, 2]);  mul_146 = None
        mul_147: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_143, 0.3535533905932738);  view_143 = None
        permute_51: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        clone_17: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_144: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_17, _shape_param_146);  clone_17 = _shape_param_146 = None
        view_145: "bf16[32768, 768]" = torch.ops.aten.view.default(view_144, _shape_param_147);  view_144 = _shape_param_147 = None
        mm_58: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_145, arg131_1);  arg131_1 = None
        permute_52: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_145, [1, 0])
        mm_59: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_52, arg132_1);  permute_52 = None
        sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_145, [0], True, dtype = torch.float32);  view_145 = None
        view_146: "f32[768]" = torch.ops.aten.view.default(sum_80, _shape_param_148);  sum_80 = _shape_param_148 = None
        convert_element_type_168: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_146, torch.bfloat16);  view_146 = None
        view_147: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_58, _shape_param_149);  mm_58 = _shape_param_149 = None
        convert_element_type_169: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_147, torch.float32);  view_147 = None
        add_30: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_141, convert_element_type_169);  mul_141 = convert_element_type_169 = None
        convert_element_type_170: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_171: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_168, torch.float32);  convert_element_type_168 = None
        permute_53: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_50, [0, 2, 1, 3]);  permute_50 = None
        view_148: "bf16[256, 128, 768]" = torch.ops.aten.view.default(permute_53, _shape_param_150);  permute_53 = _shape_param_150 = None
        clone_18: "bf16[256, 128, 768]" = torch.ops.aten.clone.default(view_148, memory_format = torch.contiguous_format);  view_148 = None
        view_149: "bf16[32768, 768]" = torch.ops.aten.view.default(clone_18, _shape_param_151);  clone_18 = _shape_param_151 = None
        mm_60: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_149, arg133_1);  arg133_1 = None
        permute_54: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_149, [1, 0])
        mm_61: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_54, arg132_1);  permute_54 = None
        sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True, dtype = torch.float32);  view_149 = None
        view_150: "f32[768]" = torch.ops.aten.view.default(sum_81, _shape_param_152);  sum_81 = _shape_param_152 = None
        convert_element_type_172: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_150, torch.bfloat16);  view_150 = None
        view_151: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_60, _shape_param_153);  mm_60 = _shape_param_153 = None
        convert_element_type_173: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        add_31: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_30, convert_element_type_173);  add_30 = convert_element_type_173 = None
        convert_element_type_174: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_175: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_172, torch.float32);  convert_element_type_172 = None
        permute_55: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_147, [0, 2, 1, 3]);  mul_147 = None
        clone_19: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_152: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_19, _shape_param_154);  clone_19 = _shape_param_154 = None
        view_153: "bf16[32768, 768]" = torch.ops.aten.view.default(view_152, _shape_param_155);  view_152 = _shape_param_155 = None
        mm_62: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_153, arg134_1);  arg134_1 = None
        permute_56: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_153, [1, 0])
        mm_63: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_56, arg132_1);  permute_56 = arg132_1 = None
        sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_153, [0], True, dtype = torch.float32);  view_153 = None
        view_154: "f32[768]" = torch.ops.aten.view.default(sum_82, _shape_param_156);  sum_82 = _shape_param_156 = None
        convert_element_type_176: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_154, torch.bfloat16);  view_154 = None
        view_155: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_62, _shape_param_157);  mm_62 = _shape_param_157 = None
        convert_element_type_177: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None
        add_32: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_31, convert_element_type_177);  add_31 = convert_element_type_177 = None
        convert_element_type_178: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_179: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_176, torch.float32);  convert_element_type_176 = None
        mul_148: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_32, arg135_1);  arg135_1 = None
        mul_149: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_148, 768)
        sum_83: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
        mul_150: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_148, arg136_1);  mul_148 = None
        sum_84: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
        mul_151: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg136_1, sum_84);  sum_84 = None
        sub_26: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_149, sum_83);  mul_149 = sum_83 = None
        sub_27: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_151);  sub_26 = mul_151 = None
        mul_152: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg137_1, sub_27);  arg137_1 = sub_27 = None
        mul_153: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_32, arg136_1);  arg136_1 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
        sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_32, [0, 1]);  add_32 = None
        convert_element_type_180: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16)
        convert_element_type_181: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.bfloat16);  arg138_1 = None
        mul_154: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_181, 1.1111111111111112);  convert_element_type_181 = None
        mul_155: "bf16[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_180, mul_154);  convert_element_type_180 = mul_154 = None
        view_156: "bf16[32768, 768]" = torch.ops.aten.view.default(mul_155, _shape_param_158);  mul_155 = _shape_param_158 = None
        mm_64: "bf16[32768, 3072]" = torch.ops.aten.mm.default(view_156, arg139_1);  arg139_1 = None
        permute_57: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_65: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_57, arg140_1);  permute_57 = arg140_1 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True, dtype = torch.float32);  view_156 = None
        view_157: "f32[768]" = torch.ops.aten.view.default(sum_87, _shape_param_159);  sum_87 = _shape_param_159 = None
        convert_element_type_182: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_157, torch.bfloat16);  view_157 = None
        view_158: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(mm_64, _shape_param_160);  mm_64 = _shape_param_160 = None
        convert_element_type_183: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_184: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_182, torch.float32);  convert_element_type_182 = None
        convert_element_type_185: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        view_159: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(arg141_1, _shape_param_161);  arg141_1 = _shape_param_161 = None
        convert_element_type_186: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        mul_156: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_186, 0.7071067811865476)
        erf_6: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_156);  mul_156 = None
        add_33: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_157: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
        mul_158: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_186, convert_element_type_186)
        mul_159: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_158, -0.5);  mul_158 = None
        exp_7: "f32[256, 128, 3072]" = torch.ops.aten.exp.default(mul_159);  mul_159 = None
        mul_160: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_161: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_186, mul_160);  convert_element_type_186 = mul_160 = None
        add_34: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(mul_157, mul_161);  mul_157 = mul_161 = None
        mul_162: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_185, add_34);  convert_element_type_185 = add_34 = None
        convert_element_type_187: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None
        view_160: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_187, _shape_param_162);  convert_element_type_187 = _shape_param_162 = None
        mm_66: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_160, arg142_1);  arg142_1 = None
        permute_58: "bf16[3072, 32768]" = torch.ops.aten.permute.default(view_160, [1, 0])
        mm_67: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_58, arg143_1);  permute_58 = arg143_1 = None
        sum_88: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True, dtype = torch.float32);  view_160 = None
        view_161: "f32[3072]" = torch.ops.aten.view.default(sum_88, _shape_param_163);  sum_88 = _shape_param_163 = None
        convert_element_type_188: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_161, torch.bfloat16);  view_161 = None
        view_162: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_66, _shape_param_164);  mm_66 = _shape_param_164 = None
        convert_element_type_189: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32);  view_162 = None
        add_35: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_152, convert_element_type_189);  mul_152 = convert_element_type_189 = None
        convert_element_type_190: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_191: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_188, torch.float32);  convert_element_type_188 = None
        mul_163: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_35, arg144_1);  arg144_1 = None
        mul_164: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_163, 768)
        sum_89: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True)
        mul_165: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_163, arg145_1);  mul_163 = None
        sum_90: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True);  mul_165 = None
        mul_166: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg145_1, sum_90);  sum_90 = None
        sub_28: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_164, sum_89);  mul_164 = sum_89 = None
        sub_29: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_28, mul_166);  sub_28 = mul_166 = None
        mul_167: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg146_1, sub_29);  arg146_1 = sub_29 = None
        mul_168: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_35, arg145_1);  arg145_1 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 1]);  mul_168 = None
        sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_35, [0, 1]);  add_35 = None
        convert_element_type_192: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_167, torch.bfloat16)
        view_163: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_192, _shape_param_165);  convert_element_type_192 = _shape_param_165 = None
        mm_68: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_163, arg147_1);  arg147_1 = None
        permute_59: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_163, [1, 0])
        mm_69: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_59, arg148_1);  permute_59 = arg148_1 = None
        sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_163, [0], True, dtype = torch.float32);  view_163 = None
        view_164: "f32[768]" = torch.ops.aten.view.default(sum_93, _shape_param_166);  sum_93 = _shape_param_166 = None
        convert_element_type_193: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_164, torch.bfloat16);  view_164 = None
        view_165: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_68, _shape_param_167);  mm_68 = _shape_param_167 = None
        convert_element_type_194: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_195: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_193, torch.float32);  convert_element_type_193 = None
        view_166: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_165, _shape_param_168);  view_165 = _shape_param_168 = None
        permute_60: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        clone_20: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_167: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_169);  clone_20 = _shape_param_169 = None
        bmm_20: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(arg149_1, view_167);  arg149_1 = None
        bmm_21: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_167, arg150_1);  view_167 = arg150_1 = None
        view_168: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_170);  bmm_20 = _shape_param_170 = None
        view_169: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_21, _shape_param_171);  bmm_21 = _shape_param_171 = None
        convert_element_type_196: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.bfloat16);  arg151_1 = None
        mul_169: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_196, 1.1111111111111112);  convert_element_type_196 = None
        mul_170: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_169, mul_169);  view_169 = mul_169 = None
        convert_element_type_197: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_170, torch.float32);  mul_170 = None
        view_170: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(arg152_1, _shape_param_172);  arg152_1 = _shape_param_172 = None
        convert_element_type_198: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_170, torch.float32);  view_170 = None
        sub_30: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_198, arg153_1);  convert_element_type_198 = arg153_1 = None
        exp_8: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        div_2: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, arg154_1);  exp_8 = arg154_1 = None
        convert_element_type_199: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        full_2: "bf16[256, 12, 128, 128]" = torch.ops.aten.full.default(_shape_param_173, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_173 = None
        where_3: "bf16[256, 12, 128, 128]" = torch.ops.aten.where.self(arg155_1, full_2, convert_element_type_199);  arg155_1 = full_2 = convert_element_type_199 = None
        convert_element_type_200: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        mul_171: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_197, convert_element_type_200);  convert_element_type_197 = None
        sum_94: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [-1], True)
        neg_5: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_200);  convert_element_type_200 = None
        fma_5: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_5, sum_94, mul_171);  neg_5 = sum_94 = mul_171 = None
        convert_element_type_201: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        view_171: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(convert_element_type_201, _shape_param_174);  convert_element_type_201 = _shape_param_174 = None
        bmm_22: "bf16[3072, 64, 128]" = torch.ops.aten.bmm.default(arg156_1, view_171);  arg156_1 = None
        bmm_23: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_171, arg157_1);  view_171 = arg157_1 = None
        view_172: "bf16[256, 12, 64, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_175);  bmm_22 = _shape_param_175 = None
        view_173: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_176);  bmm_23 = _shape_param_176 = None
        mul_172: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(view_172, 0.3535533905932738);  view_172 = None
        permute_61: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(mul_172, [0, 1, 3, 2]);  mul_172 = None
        mul_173: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(view_173, 0.3535533905932738);  view_173 = None
        permute_62: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_21: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_174: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_21, _shape_param_177);  clone_21 = _shape_param_177 = None
        view_175: "bf16[32768, 768]" = torch.ops.aten.view.default(view_174, _shape_param_178);  view_174 = _shape_param_178 = None
        mm_70: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_175, arg158_1);  arg158_1 = None
        permute_63: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_175, [1, 0])
        mm_71: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_63, arg159_1);  permute_63 = None
        sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_175, [0], True, dtype = torch.float32);  view_175 = None
        view_176: "f32[768]" = torch.ops.aten.view.default(sum_95, _shape_param_179);  sum_95 = _shape_param_179 = None
        convert_element_type_202: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_176, torch.bfloat16);  view_176 = None
        view_177: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_70, _shape_param_180);  mm_70 = _shape_param_180 = None
        convert_element_type_203: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        add_36: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_167, convert_element_type_203);  mul_167 = convert_element_type_203 = None
        convert_element_type_204: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_205: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_202, torch.float32);  convert_element_type_202 = None
        permute_64: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(permute_61, [0, 2, 1, 3]);  permute_61 = None
        view_178: "bf16[256, 128, 768]" = torch.ops.aten.view.default(permute_64, _shape_param_181);  permute_64 = _shape_param_181 = None
        clone_22: "bf16[256, 128, 768]" = torch.ops.aten.clone.default(view_178, memory_format = torch.contiguous_format);  view_178 = None
        view_179: "bf16[32768, 768]" = torch.ops.aten.view.default(clone_22, _shape_param_182);  clone_22 = _shape_param_182 = None
        mm_72: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_179, arg160_1);  arg160_1 = None
        permute_65: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_179, [1, 0])
        mm_73: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_65, arg159_1);  permute_65 = None
        sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True, dtype = torch.float32);  view_179 = None
        view_180: "f32[768]" = torch.ops.aten.view.default(sum_96, _shape_param_183);  sum_96 = _shape_param_183 = None
        convert_element_type_206: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_180, torch.bfloat16);  view_180 = None
        view_181: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_72, _shape_param_184);  mm_72 = _shape_param_184 = None
        convert_element_type_207: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_181, torch.float32);  view_181 = None
        add_37: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_36, convert_element_type_207);  add_36 = convert_element_type_207 = None
        convert_element_type_208: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_209: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_206, torch.float32);  convert_element_type_206 = None
        permute_66: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(mul_173, [0, 2, 1, 3]);  mul_173 = None
        clone_23: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
        view_182: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_23, _shape_param_185);  clone_23 = _shape_param_185 = None
        view_183: "bf16[32768, 768]" = torch.ops.aten.view.default(view_182, _shape_param_186);  view_182 = _shape_param_186 = None
        mm_74: "bf16[32768, 768]" = torch.ops.aten.mm.default(view_183, arg161_1);  arg161_1 = None
        permute_67: "bf16[768, 32768]" = torch.ops.aten.permute.default(view_183, [1, 0])
        mm_75: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_67, arg159_1);  permute_67 = arg159_1 = None
        sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True, dtype = torch.float32);  view_183 = None
        view_184: "f32[768]" = torch.ops.aten.view.default(sum_97, _shape_param_187);  sum_97 = _shape_param_187 = None
        convert_element_type_210: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_184, torch.bfloat16);  view_184 = None
        view_185: "bf16[256, 128, 768]" = torch.ops.aten.view.default(mm_74, _shape_param_188);  mm_74 = _shape_param_188 = None
        convert_element_type_211: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_185, torch.float32);  view_185 = None
        add_38: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_37, convert_element_type_211);  add_37 = convert_element_type_211 = None
        convert_element_type_212: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_213: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_210, torch.float32);  convert_element_type_210 = None
        convert_element_type_214: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.float32);  arg162_1 = None
        mul_174: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_214, 1.1111111111111112);  convert_element_type_214 = None
        mul_175: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_38, mul_174);  add_38 = mul_174 = None
        mul_176: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_175, arg163_1);  arg163_1 = None
        mul_177: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_176, 768)
        sum_98: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True)
        add_39: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(arg164_1, arg165_1);  arg164_1 = arg165_1 = None
        sub_31: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(add_39, arg166_1);  add_39 = arg166_1 = None
        mul_178: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_31, arg167_1);  sub_31 = None
        mul_179: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_176, mul_178);  mul_176 = None
        sum_99: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True);  mul_179 = None
        mul_180: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_178, sum_99);  sum_99 = None
        sub_32: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_177, sum_98);  mul_177 = sum_98 = None
        sub_33: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_180);  sub_32 = mul_180 = None
        div_3: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(arg167_1, 768);  arg167_1 = None
        mul_181: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_33);  div_3 = sub_33 = None
        mul_182: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_175, mul_178);  mul_178 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1]);  mul_182 = None
        sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
        sum_102: "f32[1, 128, 768]" = torch.ops.aten.sum.dim_IntList(mul_181, [0], True, dtype = torch.float32)
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg168_1, 1, 0, 128);  arg168_1 = None
        ge: "b8[1, 128]" = torch.ops.aten.ge.Scalar(slice_1, 0)
        lt: "b8[1, 128]" = torch.ops.aten.lt.Scalar(slice_1, 512)
        bitwise_and: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_1: "b8[1, 128]" = torch.ops.aten.ne.Scalar(slice_1, -1)
        bitwise_and_1: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_1);  bitwise_and = ne_1 = None
        unsqueeze_1: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_3: "f32[512, 768]" = torch.ops.aten.full.default(_shape_param_189, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_189 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_3, unsqueeze_1, [slice_1], sum_102);  full_3 = unsqueeze_1 = slice_1 = sum_102 = None
        ge_1: "b8[256, 128]" = torch.ops.aten.ge.Scalar(arg169_1, 0)
        lt_1: "b8[256, 128]" = torch.ops.aten.lt.Scalar(arg169_1, 30522)
        bitwise_and_2: "b8[256, 128]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_2: "b8[256, 128]" = torch.ops.aten.ne.Scalar(arg169_1, 0)
        bitwise_and_3: "b8[256, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_2);  bitwise_and_2 = ne_2 = None
        unsqueeze_2: "b8[256, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_4: "f32[30522, 768]" = torch.ops.aten.full.default(_shape_param_190, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_190 = None
        _unsafe_masked_index_put_accumulate_1: "f32[30522, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_2, [arg169_1], mul_181);  full_4 = unsqueeze_2 = arg169_1 = mul_181 = None
        add_40: "f32[30522, 768]" = torch.ops.aten.add.Tensor(convert_element_type_8, _unsafe_masked_index_put_accumulate_1);  convert_element_type_8 = _unsafe_masked_index_put_accumulate_1 = None
        return (convert_element_type_9, sum_5, sum_6, convert_element_type_18, convert_element_type_19, sum_10, sum_11, convert_element_type_23, convert_element_type_24, convert_element_type_30, convert_element_type_31, sum_16, sum_17, convert_element_type_34, convert_element_type_35, convert_element_type_42, convert_element_type_43, convert_element_type_46, convert_element_type_47, convert_element_type_50, convert_element_type_51, sum_25, sum_26, convert_element_type_55, convert_element_type_56, convert_element_type_62, convert_element_type_63, sum_31, sum_32, convert_element_type_66, convert_element_type_67, convert_element_type_74, convert_element_type_75, convert_element_type_78, convert_element_type_79, convert_element_type_82, convert_element_type_83, sum_40, sum_41, convert_element_type_87, convert_element_type_88, convert_element_type_94, convert_element_type_95, sum_46, sum_47, convert_element_type_98, convert_element_type_99, convert_element_type_106, convert_element_type_107, convert_element_type_110, convert_element_type_111, convert_element_type_114, convert_element_type_115, sum_55, sum_56, convert_element_type_119, convert_element_type_120, convert_element_type_126, convert_element_type_127, sum_61, sum_62, convert_element_type_130, convert_element_type_131, convert_element_type_138, convert_element_type_139, convert_element_type_142, convert_element_type_143, convert_element_type_146, convert_element_type_147, sum_70, sum_71, convert_element_type_151, convert_element_type_152, convert_element_type_158, convert_element_type_159, sum_76, sum_77, convert_element_type_162, convert_element_type_163, convert_element_type_170, convert_element_type_171, convert_element_type_174, convert_element_type_175, convert_element_type_178, convert_element_type_179, sum_85, sum_86, convert_element_type_183, convert_element_type_184, convert_element_type_190, convert_element_type_191, sum_91, sum_92, convert_element_type_194, convert_element_type_195, convert_element_type_204, convert_element_type_205, convert_element_type_208, convert_element_type_209, convert_element_type_212, convert_element_type_213, sum_100, sum_101, _unsafe_masked_index_put_accumulate, add_40)



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
