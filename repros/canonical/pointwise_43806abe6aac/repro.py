"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g20
Pattern hash: 43806abe6aac
Shape hash: 3e2ab305
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_48: "f16[8, 1000]", addmm_49: "f16[8, 1000]", permute_74: "f16[768, 1000]", permute_73: "f16[768, 1000]", rsqrt_24: "f32[8, 198, 1]", permute_72: "f16[3072, 768]", permute_71: "f16[768, 3072]", rsqrt_23: "f32[8, 198, 1]", permute_70: "f16[768, 768]", permute_67: "f16[768, 2304]", rsqrt_22: "f32[8, 198, 1]", permute_66: "f16[3072, 768]", permute_65: "f16[768, 3072]", rsqrt_21: "f32[8, 198, 1]", permute_64: "f16[768, 768]", permute_61: "f16[768, 2304]", rsqrt_20: "f32[8, 198, 1]", permute_60: "f16[3072, 768]", permute_59: "f16[768, 3072]", rsqrt_19: "f32[8, 198, 1]", permute_58: "f16[768, 768]", permute_55: "f16[768, 2304]", rsqrt_18: "f32[8, 198, 1]", permute_54: "f16[3072, 768]", permute_53: "f16[768, 3072]", rsqrt_17: "f32[8, 198, 1]", permute_52: "f16[768, 768]", permute_49: "f16[768, 2304]", rsqrt_16: "f32[8, 198, 1]", permute_48: "f16[3072, 768]", permute_47: "f16[768, 3072]", rsqrt_15: "f32[8, 198, 1]", permute_46: "f16[768, 768]", permute_43: "f16[768, 2304]", rsqrt_14: "f32[8, 198, 1]", permute_42: "f16[3072, 768]", permute_41: "f16[768, 3072]", rsqrt_13: "f32[8, 198, 1]", permute_40: "f16[768, 768]", permute_37: "f16[768, 2304]", rsqrt_12: "f32[8, 198, 1]", permute_36: "f16[3072, 768]", permute_35: "f16[768, 3072]", rsqrt_11: "f32[8, 198, 1]", permute_34: "f16[768, 768]", permute_31: "f16[768, 2304]", rsqrt_10: "f32[8, 198, 1]", permute_30: "f16[3072, 768]", permute_29: "f16[768, 3072]", rsqrt_9: "f32[8, 198, 1]", permute_28: "f16[768, 768]", permute_25: "f16[768, 2304]", rsqrt_8: "f32[8, 198, 1]", permute_24: "f16[3072, 768]", permute_23: "f16[768, 3072]", rsqrt_7: "f32[8, 198, 1]", permute_22: "f16[768, 768]", permute_19: "f16[768, 2304]", rsqrt_6: "f32[8, 198, 1]", permute_18: "f16[3072, 768]", permute_17: "f16[768, 3072]", rsqrt_5: "f32[8, 198, 1]", permute_16: "f16[768, 768]", permute_13: "f16[768, 2304]", rsqrt_4: "f32[8, 198, 1]", permute_12: "f16[3072, 768]", permute_11: "f16[768, 3072]", rsqrt_3: "f32[8, 198, 1]", permute_10: "f16[768, 768]", permute_7: "f16[768, 2304]", rsqrt_2: "f32[8, 198, 1]", permute_6: "f16[3072, 768]", permute_5: "f16[768, 3072]", rsqrt_1: "f32[8, 198, 1]", permute_4: "f16[768, 768]", permute_1: "f16[768, 2304]"):
        # No stacktrace found for following nodes
        add_tensor: "f16[8, 1000]" = torch.ops.aten.add.Tensor(addmm_48, addmm_49);  addmm_48 = addmm_49 = None
        div_tensor: "f16[8, 1000]" = torch.ops.aten.div.Tensor(add_tensor, 2);  add_tensor = None
        permute_default: "f16[1000, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_default_1: "f16[1000, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        div_tensor_1: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        permute_default_2: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        permute_default_3: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        div_tensor_2: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        permute_default_4: "f16[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        permute_default_5: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        div_tensor_3: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        permute_default_6: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        permute_default_7: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        div_tensor_4: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        permute_default_8: "f16[768, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        permute_default_9: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        div_tensor_5: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        permute_default_10: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        permute_default_11: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        div_tensor_6: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        permute_default_12: "f16[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_default_13: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        div_tensor_7: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        permute_default_14: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_default_15: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        div_tensor_8: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        permute_default_16: "f16[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_default_17: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        div_tensor_9: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        permute_default_18: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        permute_default_19: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        div_tensor_10: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        permute_default_20: "f16[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_default_21: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        div_tensor_11: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        permute_default_22: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        permute_default_23: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        div_tensor_12: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        permute_default_24: "f16[768, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        permute_default_25: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        div_tensor_13: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        permute_default_26: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        permute_default_27: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        div_tensor_14: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_default_28: "f16[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_default_29: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        div_tensor_15: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        permute_default_30: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_default_31: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        div_tensor_16: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        permute_default_32: "f16[768, 768]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        permute_default_33: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        div_tensor_17: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        permute_default_34: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_default_35: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        div_tensor_18: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_default_36: "f16[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_default_37: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        div_tensor_19: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        permute_default_38: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        permute_default_39: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        div_tensor_20: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_default_40: "f16[768, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        permute_default_41: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        div_tensor_21: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        permute_default_42: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_default_43: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        div_tensor_22: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_default_44: "f16[768, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_default_45: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        div_tensor_23: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        permute_default_46: "f16[768, 3072]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        permute_default_47: "f16[3072, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        div_tensor_24: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_default_48: "f16[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_default_49: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (div_tensor, permute_default, permute_default_1, div_tensor_1, permute_default_2, permute_default_3, div_tensor_2, permute_default_4, permute_default_5, div_tensor_3, permute_default_6, permute_default_7, div_tensor_4, permute_default_8, permute_default_9, div_tensor_5, permute_default_10, permute_default_11, div_tensor_6, permute_default_12, permute_default_13, div_tensor_7, permute_default_14, permute_default_15, div_tensor_8, permute_default_16, permute_default_17, div_tensor_9, permute_default_18, permute_default_19, div_tensor_10, permute_default_20, permute_default_21, div_tensor_11, permute_default_22, permute_default_23, div_tensor_12, permute_default_24, permute_default_25, div_tensor_13, permute_default_26, permute_default_27, div_tensor_14, permute_default_28, permute_default_29, div_tensor_15, permute_default_30, permute_default_31, div_tensor_16, permute_default_32, permute_default_33, div_tensor_17, permute_default_34, permute_default_35, div_tensor_18, permute_default_36, permute_default_37, div_tensor_19, permute_default_38, permute_default_39, div_tensor_20, permute_default_40, permute_default_41, div_tensor_21, permute_default_42, permute_default_43, div_tensor_22, permute_default_44, permute_default_45, div_tensor_23, permute_default_46, permute_default_47, div_tensor_24, permute_default_48, permute_default_49)


def _default_make_inputs():
    return [
    torch.randn([8, 1000], dtype=torch.float16, device='cuda'),
    torch.randn([8, 1000], dtype=torch.float16, device='cuda'),
    torch.randn(768000, dtype=torch.float16, device='cuda').as_strided([768, 1000], [1, 768]),  # permute_74
    torch.randn(768000, dtype=torch.float16, device='cuda').as_strided([768, 1000], [1, 768]),  # permute_73
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_72
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_71
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_70
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_67
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_66
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_65
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_64
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_61
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_60
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_59
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_58
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_55
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_54
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_53
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_52
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_49
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_48
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_47
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_46
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_43
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_42
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_41
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_40
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_37
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_36
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_35
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_34
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_31
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_30
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_29
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_28
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_25
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_24
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_23
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_22
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_19
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_18
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_17
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_16
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_13
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_12
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_11
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_10
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_7
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([3072, 768], [1, 3072]),  # permute_6
    torch.randn(2359296, dtype=torch.float16, device='cuda').as_strided([768, 3072], [1, 768]),  # permute_5
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn(589824, dtype=torch.float16, device='cuda').as_strided([768, 768], [1, 768]),  # permute_4
    torch.randn(1769472, dtype=torch.float16, device='cuda').as_strided([768, 2304], [1, 768]),  # permute_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
