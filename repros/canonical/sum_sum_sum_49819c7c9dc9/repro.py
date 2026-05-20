"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 49819c7c9dc9
Shape hash: 5f9ae9bc
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 1000], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([128, 768, 7, 7], f32), T([768], f32), T([768], f32), T([128, 768, 7, 7], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([128, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)), T([192], f32), T([192], f32), T([128, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)), T([32], f32), T([32], f32), S([1000]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", sum_11: "f32[768]", squeeze_82: "f32[768]", sum_13: "f32[768]", squeeze_79: "f32[768]", sum_16: "f32[768]", squeeze_76: "f32[768]", sum_18: "f32[768]", squeeze_73: "f32[768]", sum_21: "f32[768]", squeeze_70: "f32[768]", sum_23: "f32[768]", squeeze_67: "f32[768]", sum_26: "f32[768]", squeeze_64: "f32[768]", sum_28: "f32[768]", squeeze_61: "f32[768]", sum_31: "f32[768]", squeeze_58: "f32[768]", add_203: "f32[128, 768, 7, 7]", sum_34: "f32[768]", squeeze_55: "f32[768]", mul_394: "f32[128, 768, 7, 7]", sum_37: "f32[384]", squeeze_52: "f32[384]", sum_40: "f32[384]", squeeze_49: "f32[384]", sum_42: "f32[384]", squeeze_46: "f32[384]", sum_45: "f32[384]", squeeze_43: "f32[384]", sum_47: "f32[384]", squeeze_40: "f32[384]", sum_50: "f32[384]", squeeze_37: "f32[384]", sum_52: "f32[384]", squeeze_34: "f32[384]", sum_55: "f32[384]", squeeze_31: "f32[384]", add_219: "f32[128, 384, 14, 14]", sum_58: "f32[384]", squeeze_28: "f32[384]", mul_511: "f32[128, 384, 14, 14]", sum_61: "f32[192]", squeeze_25: "f32[192]", sum_63: "f32[192]", squeeze_22: "f32[192]", sum_65: "f32[192]", squeeze_19: "f32[192]", sum_67: "f32[192]", squeeze_16: "f32[192]", sum_69: "f32[192]", squeeze_13: "f32[192]", sum_71: "f32[192]", squeeze_10: "f32[192]", sum_73: "f32[192]", squeeze_7: "f32[192]", add_254: "f32[128, 192, 28, 28]", sum_76: "f32[192]", squeeze_4: "f32[192]", mul_681: "f32[128, 192, 28, 28]", sum_79: "f32[32]", squeeze_1: "f32[32]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:474 in forward_head, code: return x if pre_logits else self.head(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:468 in forward_features, code: x = self.norm(x)
        mul_tensor: "f32[768]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_82);  sum_11 = squeeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        mul_tensor_1: "f32[768]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_79);  sum_13 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        mul_tensor_2: "f32[768]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_76);  sum_16 = squeeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        mul_tensor_3: "f32[768]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_73);  sum_18 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        mul_tensor_4: "f32[768]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_70);  sum_21 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        mul_tensor_5: "f32[768]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_67);  sum_23 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        mul_tensor_6: "f32[768]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_64);  sum_26 = squeeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        mul_tensor_7: "f32[768]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_61);  sum_28 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        mul_tensor_8: "f32[768]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_58);  sum_31 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:462 in forward_features, code: x = self.pos_drop(x + self.pos_embed3)
        sum_dim_int_list_1: "f32[1, 768, 7, 7]" = torch.ops.aten.sum.dim_IntList(add_203, [0], True);  add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        mul_tensor_9: "f32[768]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_55);  sum_34 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_394, [0, 2, 3]);  mul_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        mul_tensor_10: "f32[384]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_52);  sum_37 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_49);  sum_40 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_46);  sum_42 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_43);  sum_45 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_40);  sum_47 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_37);  sum_50 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_34);  sum_52 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_31);  sum_55 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:452 in forward_features, code: x = self.pos_drop(x + self.pos_embed2)
        sum_dim_int_list_3: "f32[1, 384, 14, 14]" = torch.ops.aten.sum.dim_IntList(add_219, [0], True);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        mul_tensor_18: "f32[384]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_28);  sum_58 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_4: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 2, 3]);  mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        mul_tensor_19: "f32[192]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_25);  sum_61 = squeeze_25 = None
        mul_tensor_20: "f32[192]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_22);  sum_63 = squeeze_22 = None
        mul_tensor_21: "f32[192]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_19);  sum_65 = squeeze_19 = None
        mul_tensor_22: "f32[192]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_16);  sum_67 = squeeze_16 = None
        mul_tensor_23: "f32[192]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_13);  sum_69 = squeeze_13 = None
        mul_tensor_24: "f32[192]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_10);  sum_71 = squeeze_10 = None
        mul_tensor_25: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_7);  sum_73 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:442 in forward_features, code: x = self.pos_drop(x + self.pos_embed1)
        sum_dim_int_list_5: "f32[1, 192, 28, 28]" = torch.ops.aten.sum.dim_IntList(add_254, [0], True);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        mul_tensor_26: "f32[192]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_4);  sum_76 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_6: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_681, [0, 2, 3]);  mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:437 in forward_features, code: x = self.stem(x)
        mul_tensor_27: "f32[32]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_1);  sum_79 = squeeze_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, sum_dim_int_list_1, mul_tensor_9, sum_dim_int_list_2, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, sum_dim_int_list_3, mul_tensor_18, sum_dim_int_list_4, mul_tensor_19, mul_tensor_20, mul_tensor_21, mul_tensor_22, mul_tensor_23, mul_tensor_24, mul_tensor_25, sum_dim_int_list_5, mul_tensor_26, sum_dim_int_list_6, mul_tensor_27)


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
