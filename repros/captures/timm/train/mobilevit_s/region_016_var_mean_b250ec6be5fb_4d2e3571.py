"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: b250ec6be5fb
Shape hash: 4d2e3571
"""
_shapes_config = "(T([32768, 192], f32), T([512, 64, 192], f32), T([192], f32), T([192], f32), T([384, 192], f32), S([512, 64, 192]), S([32768, 192]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_21: "f32[32768, 192]", add_170: "f32[512, 64, 192]", primals_210: "f32[192]", primals_211: "f32[192]", primals_212: "f32[384, 192]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(addmm_21, _shape_param_0);  addmm_21 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_tensor: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_170, reshape_default);  add_170 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 64, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 64, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_210);  mul_tensor = primals_210 = None
        add_tensor_2: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_211);  mul_tensor_1 = primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_1: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[192, 384]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([32768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192], dtype=torch.float32, device='cuda'),
    [512, 64, 192],  # _shape_param_0
    [32768, 192],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
