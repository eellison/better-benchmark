"""
Standalone repro captured via capture_hook.
Label: sum_sum_89443c70e1cf
Pattern hash: 35925ab8e4e7
Shape hash: 0f3133d9
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
_shapes_config = "(T([64, 1024, 14, 14], f32), T([64, 992, 14, 14], f32), T([64, 960, 14, 14], f32), T([64, 928, 14, 14], f32), T([64, 896, 14, 14], f32), T([64, 864, 14, 14], f32), T([64, 832, 14, 14], f32), T([64, 800, 14, 14], f32), T([64, 768, 14, 14], f32), T([64, 736, 14, 14], f32), T([64, 704, 14, 14], f32), T([64, 672, 14, 14], f32), T([64, 640, 14, 14], f32), T([64, 608, 14, 14], f32), T([64, 576, 14, 14], f32), T([64, 544, 14, 14], f32), T([64, 512, 14, 14], f32), T([64, 480, 14, 14], f32), T([64, 448, 14, 14], f32), T([64, 416, 14, 14], f32), T([64, 384, 14, 14], f32), T([], f32), T([64, 384, 14, 14], f32), T([64, 384, 14, 14], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 1024, 14, 14]", arg1_1: "f32[64, 992, 14, 14]", arg2_1: "f32[64, 960, 14, 14]", arg3_1: "f32[64, 928, 14, 14]", arg4_1: "f32[64, 896, 14, 14]", arg5_1: "f32[64, 864, 14, 14]", arg6_1: "f32[64, 832, 14, 14]", arg7_1: "f32[64, 800, 14, 14]", arg8_1: "f32[64, 768, 14, 14]", arg9_1: "f32[64, 736, 14, 14]", arg10_1: "f32[64, 704, 14, 14]", arg11_1: "f32[64, 672, 14, 14]", arg12_1: "f32[64, 640, 14, 14]", arg13_1: "f32[64, 608, 14, 14]", arg14_1: "f32[64, 576, 14, 14]", arg15_1: "f32[64, 544, 14, 14]", arg16_1: "f32[64, 512, 14, 14]", arg17_1: "f32[64, 480, 14, 14]", arg18_1: "f32[64, 448, 14, 14]", arg19_1: "f32[64, 416, 14, 14]", arg20_1: "f32[64, 384, 14, 14]", arg21_1: "f32[]", arg22_1: "f32[64, 384, 14, 14]", arg23_1: "f32[64, 384, 14, 14]", arg24_1: "f32[1, 384, 1, 1]", arg25_1: "f32[384]", arg26_1: "f32[384]"):
        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:21 in forward, code: slice_tensor: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1151, 1, 352, 384);  mul_1151 = None
        slice_tensor: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 352, 384);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:24 in forward, code: slice_tensor_1: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1169, 1, 352, 384);  mul_1169 = None
        slice_tensor_1: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 352, 384);  arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:25 in forward, code: add_tensor: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        add_tensor: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:26 in forward, code: slice_tensor_2: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1187, 1, 352, 384);  mul_1187 = None
        slice_tensor_2: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 352, 384);  arg2_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:27 in forward, code: add_tensor_1: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        add_tensor_1: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:28 in forward, code: slice_tensor_3: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1205, 1, 352, 384);  mul_1205 = None
        slice_tensor_3: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg3_1, 1, 352, 384);  arg3_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:29 in forward, code: add_tensor_2: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        add_tensor_2: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:30 in forward, code: slice_tensor_4: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1223, 1, 352, 384);  mul_1223 = None
        slice_tensor_4: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg4_1, 1, 352, 384);  arg4_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:31 in forward, code: add_tensor_3: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        add_tensor_3: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:32 in forward, code: slice_tensor_5: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1241, 1, 352, 384);  mul_1241 = None
        slice_tensor_5: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg5_1, 1, 352, 384);  arg5_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:33 in forward, code: add_tensor_4: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        add_tensor_4: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:34 in forward, code: slice_tensor_6: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1259, 1, 352, 384);  mul_1259 = None
        slice_tensor_6: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg6_1, 1, 352, 384);  arg6_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:35 in forward, code: add_tensor_5: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        add_tensor_5: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:36 in forward, code: slice_tensor_7: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1277, 1, 352, 384);  mul_1277 = None
        slice_tensor_7: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg7_1, 1, 352, 384);  arg7_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:37 in forward, code: add_tensor_6: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None
        add_tensor_6: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:38 in forward, code: slice_tensor_8: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1295, 1, 352, 384);  mul_1295 = None
        slice_tensor_8: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg8_1, 1, 352, 384);  arg8_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:39 in forward, code: add_tensor_7: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None
        add_tensor_7: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:40 in forward, code: slice_tensor_9: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1313, 1, 352, 384);  mul_1313 = None
        slice_tensor_9: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg9_1, 1, 352, 384);  arg9_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:41 in forward, code: add_tensor_8: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_7, slice_tensor_9);  add_tensor_7 = slice_tensor_9 = None
        add_tensor_8: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_7, slice_tensor_9);  add_tensor_7 = slice_tensor_9 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:42 in forward, code: slice_tensor_10: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1331, 1, 352, 384);  mul_1331 = None
        slice_tensor_10: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg10_1, 1, 352, 384);  arg10_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:43 in forward, code: add_tensor_9: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_8, slice_tensor_10);  add_tensor_8 = slice_tensor_10 = None
        add_tensor_9: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_8, slice_tensor_10);  add_tensor_8 = slice_tensor_10 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:44 in forward, code: slice_tensor_11: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1349, 1, 352, 384);  mul_1349 = None
        slice_tensor_11: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg11_1, 1, 352, 384);  arg11_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:45 in forward, code: add_tensor_10: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_9, slice_tensor_11);  add_tensor_9 = slice_tensor_11 = None
        add_tensor_10: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_9, slice_tensor_11);  add_tensor_9 = slice_tensor_11 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:46 in forward, code: slice_tensor_12: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1367, 1, 352, 384);  mul_1367 = None
        slice_tensor_12: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg12_1, 1, 352, 384);  arg12_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:47 in forward, code: add_tensor_11: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_10, slice_tensor_12);  add_tensor_10 = slice_tensor_12 = None
        add_tensor_11: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_10, slice_tensor_12);  add_tensor_10 = slice_tensor_12 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:48 in forward, code: slice_tensor_13: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1385, 1, 352, 384);  mul_1385 = None
        slice_tensor_13: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg13_1, 1, 352, 384);  arg13_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:49 in forward, code: add_tensor_12: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_11, slice_tensor_13);  add_tensor_11 = slice_tensor_13 = None
        add_tensor_12: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_11, slice_tensor_13);  add_tensor_11 = slice_tensor_13 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:50 in forward, code: slice_tensor_14: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1403, 1, 352, 384);  mul_1403 = None
        slice_tensor_14: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg14_1, 1, 352, 384);  arg14_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:51 in forward, code: add_tensor_13: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_12, slice_tensor_14);  add_tensor_12 = slice_tensor_14 = None
        add_tensor_13: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_12, slice_tensor_14);  add_tensor_12 = slice_tensor_14 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:52 in forward, code: slice_tensor_15: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1421, 1, 352, 384);  mul_1421 = None
        slice_tensor_15: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg15_1, 1, 352, 384);  arg15_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:53 in forward, code: add_tensor_14: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_13, slice_tensor_15);  add_tensor_13 = slice_tensor_15 = None
        add_tensor_14: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_13, slice_tensor_15);  add_tensor_13 = slice_tensor_15 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:54 in forward, code: slice_tensor_16: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1439, 1, 352, 384);  mul_1439 = None
        slice_tensor_16: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg16_1, 1, 352, 384);  arg16_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:55 in forward, code: add_tensor_15: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_14, slice_tensor_16);  add_tensor_14 = slice_tensor_16 = None
        add_tensor_15: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_14, slice_tensor_16);  add_tensor_14 = slice_tensor_16 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:56 in forward, code: slice_tensor_17: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1457, 1, 352, 384);  mul_1457 = None
        slice_tensor_17: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg17_1, 1, 352, 384);  arg17_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:57 in forward, code: add_tensor_16: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_15, slice_tensor_17);  add_tensor_15 = slice_tensor_17 = None
        add_tensor_16: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_15, slice_tensor_17);  add_tensor_15 = slice_tensor_17 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:58 in forward, code: slice_tensor_18: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1475, 1, 352, 384);  mul_1475 = None
        slice_tensor_18: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg18_1, 1, 352, 384);  arg18_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:59 in forward, code: add_tensor_17: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_16, slice_tensor_18);  add_tensor_16 = slice_tensor_18 = None
        add_tensor_17: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_16, slice_tensor_18);  add_tensor_16 = slice_tensor_18 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:60 in forward, code: slice_tensor_19: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1493, 1, 352, 384);  mul_1493 = None
        slice_tensor_19: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(arg19_1, 1, 352, 384);  arg19_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:61 in forward, code: add_tensor_18: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_17, slice_tensor_19);  add_tensor_17 = slice_tensor_19 = None
        add_tensor_18: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_17, slice_tensor_19);  add_tensor_17 = slice_tensor_19 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:64 in forward, code: le_scalar: "b8[64, 384, 14, 14]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        le_scalar: "b8[64, 384, 14, 14]" = torch.ops.aten.le.Scalar(arg20_1, 0);  arg20_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:65 in forward, code: where_self: "f32[64, 384, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_460);  le_scalar = full_default = getitem_460 = None
        where_self: "f32[64, 384, 14, 14]" = torch.ops.aten.where.self(le_scalar, arg21_1, arg22_1);  le_scalar = arg21_1 = arg22_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:67 in forward, code: sub_tensor: "f32[64, 384, 14, 14]" = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_1362);  cat_21 = unsqueeze_1362 = None
        sub_tensor: "f32[64, 384, 14, 14]" = torch.ops.aten.sub.Tensor(arg23_1, arg24_1);  arg23_1 = arg24_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:68 in forward, code: mul_tensor: "f32[64, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        mul_tensor: "f32[64, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:69 in forward, code: sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:74 in forward, code: mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 7.971938775510203e-05);  sum_dim_int_list_1 = None
        mul_tensor_1: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 7.971938775510203e-05);  sum_dim_int_list = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:75 in forward, code: mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(arg25_1, arg25_1)

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:76 in forward, code: mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:77 in forward, code: unsqueeze_default_3: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:78 in forward, code: unsqueeze_default_4: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_1: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:79 in forward, code: unsqueeze_default_5: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        unsqueeze_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:84 in forward, code: mul_tensor_6: "f32[64, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        mul_tensor_4: "f32[64, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_2);  sub_tensor = unsqueeze_default_2 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:85 in forward, code: sub_tensor_1: "f32[64, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_1: "f32[64, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_4);  mul_tensor_4 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:66 in forward, code: sum_dim_int_list: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sum_dim_int_list_1: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:70 in forward, code: mul_tensor_1: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 7.971938775510203e-05);  sum_dim_int_list = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 7.971938775510203e-05);  sum_dim_int_list_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:71 in forward, code: unsqueeze_default: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_3: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:72 in forward, code: unsqueeze_default_1: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_4: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:73 in forward, code: unsqueeze_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        unsqueeze_default_5: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:86 in forward, code: sub_tensor_2: "f32[64, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        sub_tensor_2: "f32[64, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_5);  sub_tensor_1 = unsqueeze_default_5 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:80 in forward, code: mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_287);  squeeze_142 = primals_287 = None
        mul_tensor_6: "f32[384]" = torch.ops.aten.mul.Tensor(arg25_1, arg26_1);  arg25_1 = arg26_1 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:81 in forward, code: unsqueeze_default_6: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_6: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:82 in forward, code: unsqueeze_default_7: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_7: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:83 in forward, code: unsqueeze_default_8: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        unsqueeze_default_8: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:87 in forward, code: mul_tensor_7: "f32[64, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        mul_tensor_7: "f32[64, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:90 in forward, code: slice_tensor_20: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 352, 384);  mul_tensor_7 = None
        slice_tensor_20: "f32[64, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 352, 384);  mul_tensor_7 = None

        # File: /tmp/scratch_space/better_benchmark/repros/canonical/sum_sum_89443c70e1cf/repro.py:91 in forward, code: add_tensor_19: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_18, slice_tensor_20);  add_tensor_18 = slice_tensor_20 = None
        add_tensor_19: "f32[64, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_18, slice_tensor_20);  add_tensor_18 = slice_tensor_20 = None
        return add_tensor_19



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
