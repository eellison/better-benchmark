"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_train
Pattern hash: 8c321013f0ba
Shape hash: eebcd2c8
"""
_shapes_config = "(T([32768, 768], f32), T([128, 256, 768], f32), T([768], f32), T([768], f32), T([1, 1, 768], f32), T([768, 768], f32), T([1536, 768], f32), S([128, 256, 768]), S([128, -1, -1]), S([128, 768]), S([32768, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f32[32768, 768]", add_80: "f32[128, 256, 768]", primals_149: "f32[768]", primals_150: "f32[768]", primals_151: "f32[1, 1, 768]", primals_152: "f32[768, 768]", primals_154: "f32[1536, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 256, 768]" = torch.ops.aten.reshape.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_tensor: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(add_80, reshape_default);  add_80 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_149);  mul_tensor = primals_149 = None
        add_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_150);  mul_tensor_1 = primals_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:101 in forward, code: q_latent = self.latent.expand(B, -1, -1)
        expand_default: "f32[128, 1, 768]" = torch.ops.aten.expand.default(primals_151, _shape_param_1);  primals_151 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        reshape_default_1: "f32[128, 768]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:104 in forward, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        reshape_default_2: "f32[32768, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default_1: "f32[768, 1536]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        return (permute_default, reshape_default_1, reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 256, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 768], dtype=torch.float32, device='cuda'),
    [128, 256, 768],  # _shape_param_0
    [128, -1, -1],  # _shape_param_1
    [128, 768],  # _shape_param_2
    [32768, 768],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
