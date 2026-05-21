"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: 47577df7f1b1
Shape hash: 8ca9880a
"""
_shapes_config = "(T([384, 64, 3, 3], f32, stride=(576, 1, 192, 64)), T([128, 384, 28, 28], f32, stride=(301056, 1, 10752, 384)), T([384, 1, 1, 1], f32), S([1, 384, 576]), S([384, 64, 3, 3]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg73_1: "f32[384, 64, 3, 3]", convolution_25: "f32[128, 384, 28, 28]", arg74_1: "f32[384, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default: "f32[384, 64, 3, 3]" = torch.ops.aten.clone.default(arg73_1, memory_format = torch.contiguous_format);  arg73_1 = None
        reshape_default: "f32[1, 384, 576]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 384, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_default: "f32[128, 384, 28, 28]" = torch.ops.aten.neg.default(convolution_25)
        exp_default: "f32[128, 384, 28, 28]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 384, 28, 28]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 384, 28, 28]" = torch.ops.aten.div.Tensor(convolution_25, add_tensor);  convolution_25 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 384, 576]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor_1: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_1: "f32[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg74_1, 0.07450538873672485);  arg74_1 = None
        reshape_default_1: "f32[384]" = torch.ops.aten.reshape.default(mul_tensor_1, [-1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 384, 576]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default);  mul_tensor = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[384, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        return (div_tensor, reshape_default_2)



def make_inputs():
    return [
    torch.randn(221184, dtype=torch.float32, device='cuda').as_strided([384, 64, 3, 3], [576, 1, 192, 64]),  # arg73_1
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 384, 28, 28], [301056, 1, 10752, 384]),  # convolution_25
    torch.randn([384, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [1, 384, 576],  # _shape_param_0
    [384, 64, 3, 3],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
