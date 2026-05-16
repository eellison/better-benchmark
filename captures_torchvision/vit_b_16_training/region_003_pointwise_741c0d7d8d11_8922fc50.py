"""
Standalone repro captured via capture_hook.
Label: vit_b_16_training
Pattern hash: 741c0d7d8d11
Shape hash: 8922fc50
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_94: "f32[4, 197, 1]", clone_93: "f32[4, 197, 768]", getitem_95: "f32[4, 197, 1]", primals_144: "f32[768]", primals_145: "f32[768]", _shape_param_0, primals_146: "f32[3072, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:117 in forward, code: y = self.ln_2(x)
        add_tensor: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_default: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(clone_93, getitem_95);  clone_93 = getitem_95 = None
        mul_tensor: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_144);  mul_tensor = primals_144 = None
        add_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_145);  mul_tensor_1 = primals_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        reshape_default: "f32[788, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [788, 768],  # _shape_param_0
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
