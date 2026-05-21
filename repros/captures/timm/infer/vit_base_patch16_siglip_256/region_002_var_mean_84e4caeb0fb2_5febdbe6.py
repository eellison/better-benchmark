"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_infer
Pattern hash: 84e4caeb0fb2
Shape hash: 5febdbe6
"""
_shapes_config = "(T([128, 768], f32), T([768], f32), T([768], f32), T([3072, 768], f32), S([128, 1, 768]), S([128, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_49: "f32[128, 768]", arg157_1: "f32[768]", arg158_1: "f32[768]", arg159_1: "f32[3072, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:118 in forward, code: x = self.proj(x)
        reshape_default: "f32[128, 1, 768]" = torch.ops.aten.reshape.default(addmm_49, _shape_param_0);  addmm_49 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 1, 768]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor: "f32[128, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg157_1);  mul_tensor = arg157_1 = None
        add_tensor_1: "f32[128, 1, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg158_1);  mul_tensor_1 = arg158_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_1: "f32[128, 768]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    [128, 1, 768],  # _shape_param_0
    [128, 768],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
