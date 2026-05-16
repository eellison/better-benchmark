"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: 741c0d7d8d11
Shape hash: a245bb83
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, cat_2: "f32[1, 7, 7, 1536]", getitem_47: "f32[1, 7, 7, 1]", getitem_46: "f32[1, 7, 7, 1]", arg151_1: "f32[1536]", arg152_1: "f32[1536]", _shape_param_0, arg153_1: "f32[768, 1536]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:84 in forward, code: x = self.norm(x)
        sub_tensor: "f32[1, 7, 7, 1536]" = torch.ops.aten.sub.Tensor(cat_2, getitem_47);  cat_2 = getitem_47 = None
        add_tensor: "f32[1, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_default: "f32[1, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 7, 7, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 7, 7, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor, arg151_1);  mul_tensor = arg151_1 = None
        add_tensor_1: "f32[1, 7, 7, 1536]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg152_1);  mul_tensor_1 = arg152_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        reshape_default: "f32[49, 1536]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[1536, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([1, 7, 7, 1536], dtype=torch.float32, device='cuda'),
    torch.randn([1, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    [49, 1536],  # _shape_param_0
    torch.randn([768, 1536], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
