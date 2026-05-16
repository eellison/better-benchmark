"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 74fe501acd45
Shape hash: fc96e264
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_56: "f32[4, 7, 7, 1]", add_128: "f32[4, 7, 7, 768]", getitem_57: "f32[4, 7, 7, 1]", primals_183: "f32[768]", primals_184: "f32[768]", _shape_param_0, primals_185: "f32[1000, 768]", rsqrt_27: "f32[4, 7, 7, 1]", view_277: "f32[96, 49, 32]", view_273: "f32[96, 49, 32]", view_274: "f32[96, 32, 49]", rsqrt_26: "f32[4, 7, 7, 1]", rsqrt_25: "f32[4, 7, 7, 1]", view_256: "f32[96, 49, 32]", view_252: "f32[96, 49, 32]", view_253: "f32[96, 32, 49]", rsqrt_22: "f32[4, 14, 14, 1]", view_233: "f32[192, 49, 32]", view_223: "f32[192, 49, 32]", view_224: "f32[192, 32, 49]", rsqrt_21: "f32[4, 14, 14, 1]", rsqrt_20: "f32[4, 14, 14, 1]", view_206: "f32[192, 49, 32]", view_202: "f32[192, 49, 32]", view_203: "f32[192, 32, 49]", rsqrt_19: "f32[4, 14, 14, 1]", rsqrt_18: "f32[4, 14, 14, 1]", view_185: "f32[192, 49, 32]", view_175: "f32[192, 49, 32]", view_176: "f32[192, 32, 49]", rsqrt_17: "f32[4, 14, 14, 1]", rsqrt_16: "f32[4, 14, 14, 1]", view_158: "f32[192, 49, 32]", view_154: "f32[192, 49, 32]", view_155: "f32[192, 32, 49]", rsqrt_15: "f32[4, 14, 14, 1]", rsqrt_14: "f32[4, 14, 14, 1]", view_137: "f32[192, 49, 32]", view_127: "f32[192, 49, 32]", view_128: "f32[192, 32, 49]", rsqrt_13: "f32[4, 14, 14, 1]", rsqrt_12: "f32[4, 14, 14, 1]", view_110: "f32[192, 49, 32]", view_106: "f32[192, 49, 32]", view_107: "f32[192, 32, 49]", rsqrt_9: "f32[4, 28, 28, 1]", view_87: "f32[384, 49, 32]", view_77: "f32[384, 49, 32]", view_78: "f32[384, 32, 49]", rsqrt_8: "f32[4, 28, 28, 1]", rsqrt_7: "f32[4, 28, 28, 1]", view_60: "f32[384, 49, 32]", view_56: "f32[384, 49, 32]", view_57: "f32[384, 32, 49]", rsqrt_4: "f32[4, 56, 56, 1]", view_37: "f32[768, 49, 32]", view_27: "f32[768, 49, 32]", view_28: "f32[768, 32, 49]", rsqrt_3: "f32[4, 56, 56, 1]", rsqrt_2: "f32[4, 56, 56, 1]", view_10: "f32[768, 49, 32]", view_6: "f32[768, 49, 32]", view_7: "f32[768, 32, 49]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:609 in forward, code: x = self.norm(x)
        add_tensor: "f32[4, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_default: "f32[4, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(add_128, getitem_57);  add_128 = getitem_57 = None
        mul_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_183);  mul_tensor = primals_183 = None
        add_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_184);  mul_tensor_1 = primals_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:321 in forward, code: return torch.permute(x, self.dims)
        permute_default: "f32[4, 768, 7, 7]" = torch.ops.aten.permute.default(add_tensor_1, [0, 3, 1, 2]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:611 in forward, code: x = self.avgpool(x)
        mean_dim: "f32[4, 768, 1, 1]" = torch.ops.aten.mean.dim(permute_default, [-1, -2], True);  permute_default = None
        as_strided_default: "f32[4, 768, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [4, 768, 1, 1], [768, 1, 768, 768]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:612 in forward, code: x = self.flatten(x)
        reshape_default: "f32[4, 768]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:613 in forward, code: x = self.head(x)
        permute_default_1: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:609 in forward, code: x = self.norm(x)
        div_tensor: "f32[4, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_1: "f32[4, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 768);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_2: "f32[96, 32, 49]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_3: "f32[96, 32, 49]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
        permute_default_4: "f32[96, 49, 32]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        div_tensor_2: "f32[4, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 768);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_3: "f32[4, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_5: "f32[96, 32, 49]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_6: "f32[96, 32, 49]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
        permute_default_7: "f32[96, 49, 32]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_4: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 384);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_8: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_9: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
        permute_default_10: "f32[192, 49, 32]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        div_tensor_5: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 384);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_6: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 384);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_11: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_12: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_202, [0, 2, 1]);  view_202 = None
        permute_default_13: "f32[192, 49, 32]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        div_tensor_7: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 384);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_8: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 384);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_14: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_15: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
        permute_default_16: "f32[192, 49, 32]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        div_tensor_9: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 384);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_10: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 384);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_17: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_18: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None
        permute_default_19: "f32[192, 49, 32]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        div_tensor_11: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 384);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_12: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 384);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_20: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_137, [0, 2, 1]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_21: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_default_22: "f32[192, 49, 32]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        div_tensor_13: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 384);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_14: "f32[4, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 384);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_23: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_24: "f32[192, 32, 49]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
        permute_default_25: "f32[192, 49, 32]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_15: "f32[4, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 192);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_26: "f32[384, 32, 49]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_27: "f32[384, 32, 49]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        permute_default_28: "f32[384, 49, 32]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        div_tensor_16: "f32[4, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 192);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_17: "f32[4, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 192);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_29: "f32[384, 32, 49]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_30: "f32[384, 32, 49]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_default_31: "f32[384, 49, 32]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_18: "f32[4, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 96);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_32: "f32[768, 32, 49]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_33: "f32[768, 32, 49]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
        permute_default_34: "f32[768, 49, 32]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        div_tensor_19: "f32[4, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 96);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        div_tensor_20: "f32[4, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 96);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        permute_default_35: "f32[768, 32, 49]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_36: "f32[768, 32, 49]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
        permute_default_37: "f32[768, 49, 32]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        return (reshape_default, permute_default_1, div_tensor, div_tensor_1, permute_default_2, permute_default_3, permute_default_4, div_tensor_2, div_tensor_3, permute_default_5, permute_default_6, permute_default_7, div_tensor_4, permute_default_8, permute_default_9, permute_default_10, div_tensor_5, div_tensor_6, permute_default_11, permute_default_12, permute_default_13, div_tensor_7, div_tensor_8, permute_default_14, permute_default_15, permute_default_16, div_tensor_9, div_tensor_10, permute_default_17, permute_default_18, permute_default_19, div_tensor_11, div_tensor_12, permute_default_20, permute_default_21, permute_default_22, div_tensor_13, div_tensor_14, permute_default_23, permute_default_24, permute_default_25, div_tensor_15, permute_default_26, permute_default_27, permute_default_28, div_tensor_16, div_tensor_17, permute_default_29, permute_default_30, permute_default_31, div_tensor_18, permute_default_32, permute_default_33, permute_default_34, div_tensor_19, div_tensor_20, permute_default_35, permute_default_36, permute_default_37)



def make_inputs():
    return [
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [4, 768],  # _shape_param_0
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([96, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([96, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([96, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([96, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([192, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([384, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([384, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 28, 28, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([384, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([384, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([768, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([768, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([768, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([768, 32, 49], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
