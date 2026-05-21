"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: a3d98e9dfaaf
Shape hash: 5a1ec9cf
"""
_shapes_config = "(T([128, 256, 1, 1], f32), T([128, 256, 56, 56], f32, stride=(802816, 1, 14336, 256)), T([128, 256, 56, 56], f32, stride=(802816, 1, 14336, 256)), T([128, 256, 1, 1], f32), T([128, 1, 1, 1], f32), S([1, 128, -1]), S([128, 256, 1, 1]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_10: "f32[128, 256, 1, 1]", convolution_8: "f32[128, 256, 56, 56]", convolution_4: "f32[128, 256, 56, 56]", primals_36: "f32[128, 256, 1, 1]", primals_37: "f32[128, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 256, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid_default);  convolution_8 = sigmoid_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_1: "f32[128, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_2: "f32[128, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 0.2);  mul_tensor_1 = None
        add_tensor: "f32[128, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_2, convolution_4);  mul_tensor_2 = convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_default: "f32[128, 256, 56, 56]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 256, 56, 56]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 256, 56, 56]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 256, 56, 56]" = torch.ops.aten.div.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None
        mul_tensor_3: "f32[128, 256, 56, 56]" = torch.ops.aten.mul.Tensor(div_tensor, 0.9805806756909201);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 128, 256]" = torch.ops.aten.reshape.default(primals_36, _shape_param_0);  primals_36 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_4: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_37, 0.11175808310508728);  primals_37 = None
        reshape_default_1: "f32[128]" = torch.ops.aten.reshape.default(mul_tensor_4, [-1]);  mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[1, 128, 256]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        mul_tensor_5: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_6: "f32[1, 128, 256]" = torch.ops.aten.mul.Tensor(mul_tensor_5, unsqueeze_default);  mul_tensor_5 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[128, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None
        return (mul_tensor_3, reshape_default_2)



def make_inputs():
    return [
    torch.randn([128, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([128, 256, 56, 56], [802816, 1, 14336, 256]),  # convolution_8
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([128, 256, 56, 56], [802816, 1, 14336, 256]),  # convolution_4
    torch.randn([128, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [1, 128, -1],  # _shape_param_0
    [128, 256, 1, 1],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
