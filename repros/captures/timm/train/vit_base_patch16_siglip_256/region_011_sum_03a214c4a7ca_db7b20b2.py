"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_train
Pattern hash: 03a214c4a7ca
Shape hash: db7b20b2
"""
_shapes_config = "(T([128, 12, 1, 64], f32), T([768, 768], f32), T([32768, 768], f32), T([768], f32), T([128, 768, 16, 16], f32, stride=(196608, 1, 12288, 768)), T([1, 256, 768], f32), T([128, 256, 1], f32), T([128, 256, 1], f32), T([128, 256, 768], f32), S([128, 1, 768]), S([128, 768]), S([128, 256, 768]), S([128, 768, 256]), S([128, 768, 16, 16]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_142: "f32[128, 12, 1, 64]", primals_152: "f32[768, 768]", mm_105: "f32[32768, 768]", primals_5: "f32[768]", convolution: "f32[128, 768, 16, 16]", primals_4: "f32[1, 256, 768]", getitem_1: "f32[128, 256, 1]", rsqrt: "f32[128, 256, 1]", add_141: "f32[128, 256, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default: "f32[128, 1, 12, 64]" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None
        reshape_default: "f32[128, 1, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        reshape_default_1: "f32[128, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_2: "f32[128, 256, 768]" = torch.ops.aten.reshape.default(mm_105, _shape_param_2);  mm_105 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(reshape_default_2, primals_5);  reshape_default_2 = primals_5 = None
        mul_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        reshape_default_3: "f32[128, 768, 256]" = torch.ops.aten.reshape.default(convolution, _shape_param_3);  convolution = _shape_param_3 = None
        permute_default_3: "f32[128, 256, 768]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add_tensor: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(permute_default_3, primals_4);  permute_default_3 = primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_5: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(add_141, mul_tensor_5);  add_141 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_default_4: "f32[128, 768, 256]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1]);  add_tensor_1 = None
        reshape_default_4: "f32[128, 768, 16, 16]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_4);  permute_default_4 = _shape_param_4 = None
        return (reshape_default_1, permute_default_2, reshape_default_4)



def make_inputs():
    return [
    torch.randn([128, 12, 1, 64], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn(25165824, dtype=torch.float32, device='cuda').as_strided([128, 768, 16, 16], [196608, 1, 12288, 768]),  # convolution
    torch.randn([1, 256, 768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 256, 768], dtype=torch.float32, device='cuda'),
    [128, 1, 768],  # _shape_param_0
    [128, 768],  # _shape_param_1
    [128, 256, 768],  # _shape_param_2
    [128, 768, 256],  # _shape_param_3
    [128, 768, 16, 16],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
