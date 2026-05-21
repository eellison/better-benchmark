"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: 0b749c27fc6f
Shape hash: 6124b10c
"""
_shapes_config = "(T([128, 1536, 7, 7], f32, stride=(75264, 1, 10752, 1536)), T([128, 1536, 14, 14], f32, stride=(301056, 1, 21504, 1536)), T([128, 1536, 14, 14], f32, stride=(301056, 1, 21504, 1536)), T([128, 1536, 14, 14], f32, stride=(301056, 1, 21504, 1536)), T([128, 1536, 14, 14], f32, stride=(301056, 1, 21504, 1536)), T([128, 1536, 1, 1], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_171: "f32[128, 1536, 7, 7]", mul_165: "f32[128, 1536, 14, 14]", getitem_168: "f32[128, 1536, 14, 14]", add_90: "f32[128, 1536, 14, 14]", convolution_58: "f32[128, 1536, 14, 14]", convolution_60: "f32[128, 1536, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_backward_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(getitem_171, mul_165, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_171 = mul_165 = None
        add_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(getitem_168, avg_pool2d_backward_default);  getitem_168 = avg_pool2d_backward_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor, 0.8980265101338745);  add_tensor = None
        neg_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.neg.default(add_90)
        exp_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_2: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = None
        sub_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_3: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_90, sub_tensor);  add_90 = sub_tensor = None
        add_tensor_2: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_3, 1);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_2);  mul_tensor_2 = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_5: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.2);  mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_6: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_5, 2.0);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_7: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_6, convolution_58);  mul_tensor_6 = convolution_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        sum_dim_int_list: "f32[128, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2, 3], True);  mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_tensor_1: "f32[128, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_8: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_9: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_8);  sum_dim_int_list = mul_tensor_8 = None
        return mul_tensor_9



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 1536, 7, 7], [75264, 1, 10752, 1536]),  # getitem_171
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 1536, 14, 14], [301056, 1, 21504, 1536]),  # mul_165
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 1536, 14, 14], [301056, 1, 21504, 1536]),  # getitem_168
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 1536, 14, 14], [301056, 1, 21504, 1536]),  # add_90
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 1536, 14, 14], [301056, 1, 21504, 1536]),  # convolution_58
    torch.randn([128, 1536, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
