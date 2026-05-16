"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: c686d1c23eab
Shape hash: e431db13
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[4, 56, 56, 1]", clone: "f32[4, 56, 56, 96]", getitem_1: "f32[4, 56, 56, 1]", primals_4: "f32[96]", primals_5: "f32[96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:608 in forward, code: x = self.features(x)
        add_tensor: "f32[4, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[4, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
        mul_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_4);  mul_tensor = primals_4 = None
        add_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_5);  mul_tensor_1 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [3], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem_2: "f32[4, 56, 56, 1]" = var_mean_correction[0]
        getitem_3: "f32[4, 56, 56, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem_2, getitem_3)



def make_inputs():
    return [
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
