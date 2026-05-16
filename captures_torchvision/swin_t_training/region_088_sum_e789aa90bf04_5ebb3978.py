"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: e789aa90bf04
Shape hash: 5ebb3978
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_19: "f32[196, 768]", _shape_param_0, _shape_param_1, _shape_param_2, primals_155: "f32[768]", mm_2: "f32[196, 768]", _shape_param_3, getitem_49: "f32[4, 7, 7, 1]", rsqrt_24: "f32[4, 7, 7, 1]", add_139: "f32[4, 7, 7, 768]", _shape_param_4, primals_154: "f32[768, 1536]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default: "f32[4, 49, 768]" = torch.ops.aten.reshape.default(mm_19, _shape_param_0);  mm_19 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:172 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        reshape_default_1: "f32[4, 1, 1, 7, 7, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[4, 1, 7, 1, 7, 768]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:171 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        reshape_default_2: "f32[4, 7, 7, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(reshape_default_2, primals_155);  reshape_default_2 = primals_155 = None
        mul_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        reshape_default_3: "f32[4, 7, 7, 768]" = torch.ops.aten.reshape.default(mm_2, _shape_param_3);  mm_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        sub_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(reshape_default_3, getitem_49);  reshape_default_3 = getitem_49 = None
        mul_tensor_2: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_24);  sub_tensor = None
        mul_tensor_3: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[4, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        mul_tensor_5: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.add.Tensor(add_139, mul_tensor_5);  add_139 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:85 in forward, code: x = self.reduction(x)  # ... H/2 W/2 2*C
        clone_default: "f32[4, 7, 7, 768]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format);  add_tensor = None
        reshape_default_4: "f32[196, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        permute_default_1: "f32[1536, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        permute_default_2: "f32[768, 1536]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_4, permute_default_2)



def make_inputs():
    return [
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    [4, 49, 768],  # _shape_param_0
    [4, 1, 1, 7, 7, 768],  # _shape_param_1
    [4, 7, 7, 768],  # _shape_param_2
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    [4, 7, 7, 768],  # _shape_param_3
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn(150528, dtype=torch.float32, device='cuda').as_strided([4, 7, 7, 768], [37632, 7, 1, 49]),  # add_139
    [196, 768],  # _shape_param_4
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
