"""
Standalone repro captured via capture_hook.
Label: vgg16_training
Pattern hash: caf00befef88
Shape hash: 2c09d53a
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_8: "f32[4, 512, 7, 7]", _shape_param_0, primals_28: "f32[4096, 25088]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:67 in forward, code: x = self.avgpool(x)
        _adaptive_avg_pool2d_default: "f32[4, 512, 7, 7]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem_8, [7, 7]);  getitem_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:68 in forward, code: x = torch.flatten(x, 1)
        reshape_default: "f32[4, 25088]" = torch.ops.aten.reshape.default(_adaptive_avg_pool2d_default, _shape_param_0);  _adaptive_avg_pool2d_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        permute_default: "f32[25088, 4096]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([4, 512, 7, 7], dtype=torch.float32, device='cuda'),
    [4, 25088],  # _shape_param_0
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
