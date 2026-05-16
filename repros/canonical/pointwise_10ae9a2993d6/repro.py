"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_training
Pattern hash: 10ae9a2993d6
Shape hash: 9f872bbf
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_51: "f32[32, 768]", _shape_param_0, view_129: "f32[32, 1, 768]", add_84: "f32[32, 256, 768]", getitem_133: "f32[32, 256, 1]", rsqrt_24: "f32[32, 256, 1]", add_80: "f32[32, 256, 768]", getitem_131: "f32[32, 256, 1]", rsqrt_23: "f32[32, 256, 1]", add_77: "f32[32, 256, 768]", getitem_122: "f32[32, 256, 1]", rsqrt_22: "f32[32, 256, 1]", add_73: "f32[32, 256, 768]", getitem_120: "f32[32, 256, 1]", rsqrt_21: "f32[32, 256, 1]", add_70: "f32[32, 256, 768]", getitem_111: "f32[32, 256, 1]", rsqrt_20: "f32[32, 256, 1]", add_66: "f32[32, 256, 768]", getitem_109: "f32[32, 256, 1]", rsqrt_19: "f32[32, 256, 1]", add_63: "f32[32, 256, 768]", getitem_100: "f32[32, 256, 1]", rsqrt_18: "f32[32, 256, 1]", add_59: "f32[32, 256, 768]", getitem_98: "f32[32, 256, 1]", rsqrt_17: "f32[32, 256, 1]", add_56: "f32[32, 256, 768]", getitem_89: "f32[32, 256, 1]", rsqrt_16: "f32[32, 256, 1]", add_52: "f32[32, 256, 768]", getitem_87: "f32[32, 256, 1]", rsqrt_15: "f32[32, 256, 1]", add_49: "f32[32, 256, 768]", getitem_78: "f32[32, 256, 1]", rsqrt_14: "f32[32, 256, 1]", add_45: "f32[32, 256, 768]", getitem_76: "f32[32, 256, 1]", rsqrt_13: "f32[32, 256, 1]", add_42: "f32[32, 256, 768]", getitem_67: "f32[32, 256, 1]", rsqrt_12: "f32[32, 256, 1]", add_38: "f32[32, 256, 768]", getitem_65: "f32[32, 256, 1]", rsqrt_11: "f32[32, 256, 1]", add_35: "f32[32, 256, 768]", getitem_56: "f32[32, 256, 1]", rsqrt_10: "f32[32, 256, 1]", add_31: "f32[32, 256, 768]", getitem_54: "f32[32, 256, 1]", rsqrt_9: "f32[32, 256, 1]", add_28: "f32[32, 256, 768]", getitem_45: "f32[32, 256, 1]", rsqrt_8: "f32[32, 256, 1]", add_24: "f32[32, 256, 768]", getitem_43: "f32[32, 256, 1]", rsqrt_7: "f32[32, 256, 1]", add_21: "f32[32, 256, 768]", getitem_34: "f32[32, 256, 1]", rsqrt_6: "f32[32, 256, 1]", add_17: "f32[32, 256, 768]", getitem_32: "f32[32, 256, 1]", rsqrt_5: "f32[32, 256, 1]", add_14: "f32[32, 256, 768]", getitem_23: "f32[32, 256, 1]", rsqrt_4: "f32[32, 256, 1]", add_10: "f32[32, 256, 768]", getitem_21: "f32[32, 256, 1]", rsqrt_3: "f32[32, 256, 1]", add_7: "f32[32, 256, 768]", getitem_12: "f32[32, 256, 1]", rsqrt_2: "f32[32, 256, 1]", add_3: "f32[32, 256, 768]", getitem_10: "f32[32, 256, 1]", rsqrt_1: "f32[32, 256, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[32, 1, 768]" = torch.ops.aten.reshape.default(addmm_51, _shape_param_0);  addmm_51 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:122 in forward, code: x = x + self.mlp(self.norm(x))
        add_tensor: "f32[32, 1, 768]" = torch.ops.aten.add.Tensor(view_129, reshape_default);  view_129 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:126 in forward, code: x = x[:, 0]
        select_int: "f32[32, 768]" = torch.ops.aten.select.int(add_tensor, 1, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_133);  add_84 = getitem_133 = None
        mul_tensor: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_24);  sub_tensor = None
        div_tensor: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        sub_tensor_1: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_131);  add_80 = getitem_131 = None
        mul_tensor_1: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_23);  sub_tensor_1 = None
        div_tensor_1: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        sub_tensor_2: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_122);  add_77 = getitem_122 = None
        mul_tensor_2: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_22);  sub_tensor_2 = None
        div_tensor_2: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        sub_tensor_3: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_120);  add_73 = getitem_120 = None
        mul_tensor_3: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_21);  sub_tensor_3 = None
        div_tensor_3: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        sub_tensor_4: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_111);  add_70 = getitem_111 = None
        mul_tensor_4: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_20);  sub_tensor_4 = None
        div_tensor_4: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        sub_tensor_5: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_109);  add_66 = getitem_109 = None
        mul_tensor_5: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_19);  sub_tensor_5 = None
        div_tensor_5: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        sub_tensor_6: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_100);  add_63 = getitem_100 = None
        mul_tensor_6: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_6, rsqrt_18);  sub_tensor_6 = None
        div_tensor_6: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        sub_tensor_7: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_98);  add_59 = getitem_98 = None
        mul_tensor_7: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_7, rsqrt_17);  sub_tensor_7 = None
        div_tensor_7: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        sub_tensor_8: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_89);  add_56 = getitem_89 = None
        mul_tensor_8: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_8, rsqrt_16);  sub_tensor_8 = None
        div_tensor_8: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        sub_tensor_9: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_87);  add_52 = getitem_87 = None
        mul_tensor_9: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_9, rsqrt_15);  sub_tensor_9 = None
        div_tensor_9: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        sub_tensor_10: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_78);  add_49 = getitem_78 = None
        mul_tensor_10: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_10, rsqrt_14);  sub_tensor_10 = None
        div_tensor_10: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        sub_tensor_11: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_76);  add_45 = getitem_76 = None
        mul_tensor_11: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_11, rsqrt_13);  sub_tensor_11 = None
        div_tensor_11: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        sub_tensor_12: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_67);  add_42 = getitem_67 = None
        mul_tensor_12: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_12, rsqrt_12);  sub_tensor_12 = None
        div_tensor_12: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        sub_tensor_13: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_65);  add_38 = getitem_65 = None
        mul_tensor_13: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_13, rsqrt_11);  sub_tensor_13 = None
        div_tensor_13: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        sub_tensor_14: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_56);  add_35 = getitem_56 = None
        mul_tensor_14: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_14, rsqrt_10);  sub_tensor_14 = None
        div_tensor_14: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        sub_tensor_15: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_54);  add_31 = getitem_54 = None
        mul_tensor_15: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_15, rsqrt_9);  sub_tensor_15 = None
        div_tensor_15: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        sub_tensor_16: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_45);  add_28 = getitem_45 = None
        mul_tensor_16: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_16, rsqrt_8);  sub_tensor_16 = None
        div_tensor_16: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        sub_tensor_17: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_43);  add_24 = getitem_43 = None
        mul_tensor_17: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_17, rsqrt_7);  sub_tensor_17 = None
        div_tensor_17: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        sub_tensor_18: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_34);  add_21 = getitem_34 = None
        mul_tensor_18: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_18, rsqrt_6);  sub_tensor_18 = None
        div_tensor_18: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        sub_tensor_19: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_32);  add_17 = getitem_32 = None
        mul_tensor_19: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_19, rsqrt_5);  sub_tensor_19 = None
        div_tensor_19: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        sub_tensor_20: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_23);  add_14 = getitem_23 = None
        mul_tensor_20: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_20, rsqrt_4);  sub_tensor_20 = None
        div_tensor_20: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        sub_tensor_21: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_21);  add_10 = getitem_21 = None
        mul_tensor_21: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_21, rsqrt_3);  sub_tensor_21 = None
        div_tensor_21: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        sub_tensor_22: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_12);  add_7 = getitem_12 = None
        mul_tensor_22: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_22, rsqrt_2);  sub_tensor_22 = None
        div_tensor_22: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        sub_tensor_23: "f32[32, 256, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_10);  add_3 = getitem_10 = None
        mul_tensor_23: "f32[32, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_23, rsqrt_1);  sub_tensor_23 = None
        div_tensor_23: "f32[32, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        return (select_int, mul_tensor, div_tensor, mul_tensor_1, div_tensor_1, mul_tensor_2, div_tensor_2, mul_tensor_3, div_tensor_3, mul_tensor_4, div_tensor_4, mul_tensor_5, div_tensor_5, mul_tensor_6, div_tensor_6, mul_tensor_7, div_tensor_7, mul_tensor_8, div_tensor_8, mul_tensor_9, div_tensor_9, mul_tensor_10, div_tensor_10, mul_tensor_11, div_tensor_11, mul_tensor_12, div_tensor_12, mul_tensor_13, div_tensor_13, mul_tensor_14, div_tensor_14, mul_tensor_15, div_tensor_15, mul_tensor_16, div_tensor_16, mul_tensor_17, div_tensor_17, mul_tensor_18, div_tensor_18, mul_tensor_19, div_tensor_19, mul_tensor_20, div_tensor_20, mul_tensor_21, div_tensor_21, mul_tensor_22, div_tensor_22, mul_tensor_23, div_tensor_23)


def _default_make_inputs():
    return [
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    [32, 1, 768],  # _shape_param_0
    torch.randn([32, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_84
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_80
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_77
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_73
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_70
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_66
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_63
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_59
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_56
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_52
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_49
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_45
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_42
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_38
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_35
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_31
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_28
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_24
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_21
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_17
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_14
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_10
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_7
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 256, 768], [196608, 1, 256]),  # add_3
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
