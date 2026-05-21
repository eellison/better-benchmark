"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch14_dinov2.lvd142m_train
Pattern hash: d91702d18dd0
Shape hash: 31ca7b26
"""
_shapes_config = "(T([175360, 3072], f32), T([768, 3072], f32), S([128, 1370, 3072]), S([175360, 3072]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_46: "f32[175360, 3072]", primals_171: "f32[768, 3072]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[128, 1370, 3072]" = torch.ops.aten.reshape.default(addmm_46, _shape_param_0);  addmm_46 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_tensor: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[128, 1370, 3072]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 1370, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 1370, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_1: "f32[175360, 3072]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([175360, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    [128, 1370, 3072],  # _shape_param_0
    [175360, 3072],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
