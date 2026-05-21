"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: c12bd5f0d517
Shape hash: 19014d99
"""
_shapes_config = "(T([1536, 1536, 1, 1], f32), T([128, 384, 1, 1], f32), T([128, 1536, 14, 14], f32, stride=(301056, 1, 21504, 1536)), T([1536, 1, 1, 1], f32), S([1, 1536, -1]), S([1536, 1536, 1, 1]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg166_1: "f32[1536, 1536, 1, 1]", convolution_66: "f32[128, 384, 1, 1]", mul_165: "f32[128, 1536, 14, 14]", arg167_1: "f32[1536, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 1536, 1536]" = torch.ops.aten.reshape.default(arg166_1, _shape_param_0);  arg166_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 1536, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1536, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_default: "f32[128, 384, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_default: "f32[128, 1536, 7, 7]" = torch.ops.aten.avg_pool2d.default(mul_165, [2, 2], [2, 2], [0, 0], True, False);  mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_1: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg167_1, 0.04562504637317021);  arg167_1 = None
        reshape_default_1: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_1, [-1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default);  mul_tensor = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[1536, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        return (relu_default, avg_pool2d_default, reshape_default_2)



def make_inputs():
    return [
    torch.randn([1536, 1536, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 1536, 14, 14], [301056, 1, 21504, 1536]),  # mul_165
    torch.randn([1536, 1, 1, 1], dtype=torch.float32, device='cuda'),
    [1, 1536, -1],  # _shape_param_0
    [1536, 1536, 1, 1],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
