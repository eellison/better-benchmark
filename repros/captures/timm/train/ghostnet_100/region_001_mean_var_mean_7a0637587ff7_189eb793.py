"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 7a0637587ff7
Shape hash: 189eb793
"""
_shapes_config = "(T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([960], f32), T([960], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_93: "f32[512, 960, 7, 7]", primals_508: "f32[960]", primals_509: "f32[960]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 960, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 960, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_1);  convolution_93 = getitem_1 = None
        mul_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_508, -1);  primals_508 = None
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_509, -1);  primals_509 = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[512, 960, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[512, 960, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        as_strided_default: "f32[512, 960, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [512, 960, 1, 1], [960, 1, 960, 960]);  mean_dim = None
        return as_strided_default



def make_inputs():
    return [
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # convolution_93
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
