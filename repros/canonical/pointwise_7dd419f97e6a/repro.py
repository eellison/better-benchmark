"""
Standalone repro captured via capture_hook.
Label: vit_b_16_training
Pattern hash: 7dd419f97e6a
Shape hash: f796ef9d
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
    def forward(self, getitem_96: "f32[4, 197, 1]", clone_96: "f32[4, 197, 768]", getitem_97: "f32[4, 197, 1]", primals_150: "f32[768]", primals_151: "f32[768]", primals_152: "f32[1000, 768]", add_96: "f32[4, 197, 768]", add_92: "f32[4, 197, 768]", getitem_95: "f32[4, 197, 1]", rsqrt_23: "f32[4, 197, 1]", add_88: "f32[4, 197, 768]", getitem_89: "f32[4, 197, 1]", rsqrt_22: "f32[4, 197, 1]", add_84: "f32[4, 197, 768]", getitem_87: "f32[4, 197, 1]", rsqrt_21: "f32[4, 197, 1]", add_80: "f32[4, 197, 768]", getitem_81: "f32[4, 197, 1]", rsqrt_20: "f32[4, 197, 1]", add_76: "f32[4, 197, 768]", getitem_79: "f32[4, 197, 1]", rsqrt_19: "f32[4, 197, 1]", add_72: "f32[4, 197, 768]", getitem_73: "f32[4, 197, 1]", rsqrt_18: "f32[4, 197, 1]", add_68: "f32[4, 197, 768]", getitem_71: "f32[4, 197, 1]", rsqrt_17: "f32[4, 197, 1]", add_64: "f32[4, 197, 768]", getitem_65: "f32[4, 197, 1]", rsqrt_16: "f32[4, 197, 1]", add_60: "f32[4, 197, 768]", getitem_63: "f32[4, 197, 1]", rsqrt_15: "f32[4, 197, 1]", add_56: "f32[4, 197, 768]", getitem_57: "f32[4, 197, 1]", rsqrt_14: "f32[4, 197, 1]", add_52: "f32[4, 197, 768]", getitem_55: "f32[4, 197, 1]", rsqrt_13: "f32[4, 197, 1]", add_48: "f32[4, 197, 768]", getitem_49: "f32[4, 197, 1]", rsqrt_12: "f32[4, 197, 1]", add_44: "f32[4, 197, 768]", getitem_47: "f32[4, 197, 1]", rsqrt_11: "f32[4, 197, 1]", add_40: "f32[4, 197, 768]", getitem_41: "f32[4, 197, 1]", rsqrt_10: "f32[4, 197, 1]", add_36: "f32[4, 197, 768]", getitem_39: "f32[4, 197, 1]", rsqrt_9: "f32[4, 197, 1]", add_32: "f32[4, 197, 768]", getitem_33: "f32[4, 197, 1]", rsqrt_8: "f32[4, 197, 1]", add_28: "f32[4, 197, 768]", getitem_31: "f32[4, 197, 1]", rsqrt_7: "f32[4, 197, 1]", add_24: "f32[4, 197, 768]", getitem_25: "f32[4, 197, 1]", rsqrt_6: "f32[4, 197, 1]", add_20: "f32[4, 197, 768]", getitem_23: "f32[4, 197, 1]", rsqrt_5: "f32[4, 197, 1]", add_16: "f32[4, 197, 768]", getitem_17: "f32[4, 197, 1]", rsqrt_4: "f32[4, 197, 1]", add_12: "f32[4, 197, 768]", getitem_15: "f32[4, 197, 1]", rsqrt_3: "f32[4, 197, 1]", add_8: "f32[4, 197, 768]", getitem_9: "f32[4, 197, 1]", rsqrt_2: "f32[4, 197, 1]", add_4: "f32[4, 197, 768]", getitem_7: "f32[4, 197, 1]", rsqrt_1: "f32[4, 197, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:157 in forward, code: return self.ln(self.layers(self.dropout(input)))
        add_tensor: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_default: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(clone_96, getitem_97);  clone_96 = None
        mul_tensor: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_150);  mul_tensor = primals_150 = None
        add_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_151);  mul_tensor_1 = primals_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:301 in forward, code: x = x[:, 0]
        select_int: "f32[4, 768]" = torch.ops.aten.select.int(add_tensor_1, 1, 0);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:303 in forward, code: x = self.heads(x)
        permute_default: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:157 in forward, code: return self.ln(self.layers(self.dropout(input)))
        sub_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_97);  add_96 = getitem_97 = None
        mul_tensor_2: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = None
        div_tensor: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_2: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_95);  add_92 = getitem_95 = None
        mul_tensor_3: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_23);  sub_tensor_2 = None
        div_tensor_1: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_3: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_89);  add_88 = getitem_89 = None
        mul_tensor_4: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_22);  sub_tensor_3 = None
        div_tensor_2: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_4: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_87);  add_84 = getitem_87 = None
        mul_tensor_5: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_21);  sub_tensor_4 = None
        div_tensor_3: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_5: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_81);  add_80 = getitem_81 = None
        mul_tensor_6: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_20);  sub_tensor_5 = None
        div_tensor_4: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_6: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_79);  add_76 = getitem_79 = None
        mul_tensor_7: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_6, rsqrt_19);  sub_tensor_6 = None
        div_tensor_5: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_7: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_73);  add_72 = getitem_73 = None
        mul_tensor_8: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_7, rsqrt_18);  sub_tensor_7 = None
        div_tensor_6: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_8: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_71);  add_68 = getitem_71 = None
        mul_tensor_9: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_8, rsqrt_17);  sub_tensor_8 = None
        div_tensor_7: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_9: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_65);  add_64 = getitem_65 = None
        mul_tensor_10: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_9, rsqrt_16);  sub_tensor_9 = None
        div_tensor_8: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_10: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_63);  add_60 = getitem_63 = None
        mul_tensor_11: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_10, rsqrt_15);  sub_tensor_10 = None
        div_tensor_9: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_11: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_57);  add_56 = getitem_57 = None
        mul_tensor_12: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_11, rsqrt_14);  sub_tensor_11 = None
        div_tensor_10: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_12: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_55);  add_52 = getitem_55 = None
        mul_tensor_13: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_12, rsqrt_13);  sub_tensor_12 = None
        div_tensor_11: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_13: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_49);  add_48 = getitem_49 = None
        mul_tensor_14: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_13, rsqrt_12);  sub_tensor_13 = None
        div_tensor_12: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_14: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_47);  add_44 = getitem_47 = None
        mul_tensor_15: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_14, rsqrt_11);  sub_tensor_14 = None
        div_tensor_13: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_15: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_41);  add_40 = getitem_41 = None
        mul_tensor_16: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_15, rsqrt_10);  sub_tensor_15 = None
        div_tensor_14: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_16: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_39);  add_36 = getitem_39 = None
        mul_tensor_17: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_16, rsqrt_9);  sub_tensor_16 = None
        div_tensor_15: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_17: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_33);  add_32 = getitem_33 = None
        mul_tensor_18: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_17, rsqrt_8);  sub_tensor_17 = None
        div_tensor_16: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_18: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_31);  add_28 = getitem_31 = None
        mul_tensor_19: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_18, rsqrt_7);  sub_tensor_18 = None
        div_tensor_17: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_19: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_25);  add_24 = getitem_25 = None
        mul_tensor_20: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_19, rsqrt_6);  sub_tensor_19 = None
        div_tensor_18: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_20: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_23);  add_20 = getitem_23 = None
        mul_tensor_21: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_20, rsqrt_5);  sub_tensor_20 = None
        div_tensor_19: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_21: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_17);  add_16 = getitem_17 = None
        mul_tensor_22: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_21, rsqrt_4);  sub_tensor_21 = None
        div_tensor_20: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_22: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_15);  add_12 = getitem_15 = None
        mul_tensor_23: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_22, rsqrt_3);  sub_tensor_22 = None
        div_tensor_21: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        sub_tensor_23: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_9);  add_8 = getitem_9 = None
        mul_tensor_24: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_23, rsqrt_2);  sub_tensor_23 = None
        div_tensor_22: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        sub_tensor_24: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        mul_tensor_25: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_24, rsqrt_1);  sub_tensor_24 = None
        div_tensor_23: "f32[4, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        return (select_int, permute_default, mul_tensor_2, div_tensor, mul_tensor_3, div_tensor_1, mul_tensor_4, div_tensor_2, mul_tensor_5, div_tensor_3, mul_tensor_6, div_tensor_4, mul_tensor_7, div_tensor_5, mul_tensor_8, div_tensor_6, mul_tensor_9, div_tensor_7, mul_tensor_10, div_tensor_8, mul_tensor_11, div_tensor_9, mul_tensor_12, div_tensor_10, mul_tensor_13, div_tensor_11, mul_tensor_14, div_tensor_12, mul_tensor_15, div_tensor_13, mul_tensor_16, div_tensor_14, mul_tensor_17, div_tensor_15, mul_tensor_18, div_tensor_16, mul_tensor_19, div_tensor_17, mul_tensor_20, div_tensor_18, mul_tensor_21, div_tensor_19, mul_tensor_22, div_tensor_20, mul_tensor_23, div_tensor_21, mul_tensor_24, div_tensor_22, mul_tensor_25, div_tensor_23)


def _default_make_inputs():
    return [
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_96
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_92
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_88
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_84
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_80
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_76
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_72
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_68
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_64
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_60
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_56
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_52
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_48
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_44
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_40
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_36
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_32
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_28
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_24
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_20
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_16
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_12
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_8
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # add_4
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
