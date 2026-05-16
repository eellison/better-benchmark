"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 41cf9a7087e7
Shape hash: b6062491
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_97: "f32[12544, 96]", _shape_param_0, _shape_param_1, _shape_param_2, fmod_2: "i64[56]", primals_20: "f32[96]", mul_10: "f32[4, 56, 56, 96]", div_60: "f32[4, 56, 56, 1]", add_218: "f32[4, 56, 56, 96]", _shape_param_3, primals_18: "f32[96, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default: "f32[256, 49, 96]" = torch.ops.aten.reshape.default(mm_97, _shape_param_0);  mm_97 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:172 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        reshape_default_1: "f32[4, 8, 8, 7, 7, 96]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[4, 8, 7, 8, 7, 96]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:171 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        clone_default: "f32[4, 8, 7, 8, 7, 96]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[4, 56, 56, 96]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:167 in shifted_window_attention, code: x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        index_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.index.Tensor(reshape_default_2, [None, None, fmod_2]);  reshape_default_2 = None
        index_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.index.Tensor(index_tensor, [None, fmod_2]);  index_tensor = fmod_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(index_tensor_1, primals_20);  index_tensor_1 = primals_20 = None
        mul_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, 96)
        sum_dim_int_list: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_10);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_10, sum_dim_int_list_1);  mul_10 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(div_60, sub_tensor_1);  div_60 = sub_tensor_1 = None
        add_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.add.Tensor(add_218, mul_tensor_4);  add_218 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default_3: "f32[12544, 96]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_3);  add_tensor = _shape_param_3 = None
        permute_default_1: "f32[384, 96]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_2: "f32[96, 384]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([12544, 96], dtype=torch.float32, device='cuda'),
    [256, 49, 96],  # _shape_param_0
    [4, 8, 8, 7, 7, 96],  # _shape_param_1
    [4, 56, 56, 96],  # _shape_param_2
    torch.randint(0, 2, [56], dtype=torch.int64, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    [12544, 96],  # _shape_param_3
    torch.randn([96, 384], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
