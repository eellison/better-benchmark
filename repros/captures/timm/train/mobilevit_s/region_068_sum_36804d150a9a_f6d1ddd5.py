"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 36804d150a9a
Shape hash: f6d1ddd5
"""
_shapes_config = "(T([32768, 192], f32), T([192], f32), T([512, 64, 192], f32), T([512, 64, 1], f32), T([512, 64, 192], f32), T([192, 384], f32), S([512, 64, 192]), S([32768, 192]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_48: "f32[32768, 192]", primals_180: "f32[192]", mul_175: "f32[512, 64, 192]", div_48: "f32[512, 64, 1]", add_295: "f32[512, 64, 192]", primals_178: "f32[192, 384]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(mm_48, _shape_param_0);  mm_48 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_tensor: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(reshape_default, primals_180);  reshape_default = primals_180 = None
        mul_tensor_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, 192)
        sum_dim_int_list: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_175);  mul_tensor = None
        sum_dim_int_list_1: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_175, sum_dim_int_list_1);  mul_175 = sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(div_48, sub_tensor_1);  div_48 = sub_tensor_1 = None
        add_tensor: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_295, mul_tensor_4);  add_295 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_1: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[384, 192]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        permute_default_1: "f32[192, 384]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 192], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 384], dtype=torch.float32, device='cuda'),
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
