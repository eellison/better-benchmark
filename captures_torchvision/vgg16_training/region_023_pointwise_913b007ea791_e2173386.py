"""
Standalone repro captured via capture_hook.
Label: vgg16_training
Pattern hash: 913b007ea791
Shape hash: e2173386
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, gt: "b8[4, 4096]", mm_2: "f32[4, 4096]", le_1: "b8[4, 4096]", full_default: "f32[]", primals_28: "f32[4096, 25088]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        convert_element_type_default: "f32[4, 4096]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[4, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2.0);  convert_element_type_default = None
        mul_tensor_1: "f32[4, 4096]" = torch.ops.aten.mul.Tensor(mm_2, mul_tensor);  mm_2 = mul_tensor = None
        where_self: "f32[4, 4096]" = torch.ops.aten.where.self(le_1, full_default, mul_tensor_1);  le_1 = full_default = mul_tensor_1 = None
        permute_default: "f32[25088, 4096]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_default_1: "f32[4096, 25088]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (where_self, permute_default_1)



def make_inputs():
    return [
    torch.randint(0, 2, [4, 4096], dtype=torch.bool, device='cuda'),
    torch.randn([4, 4096], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 4096], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 25088], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
