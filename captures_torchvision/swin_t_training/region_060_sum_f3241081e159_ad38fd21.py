"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: f3241081e159
Shape hash: ad38fd21
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_101: "f32[12544, 96]", _shape_param_0, primals_14: "f32[96]", mul_5: "f32[4, 56, 56, 96]", div_61: "f32[4, 56, 56, 1]", add_225: "f32[4, 56, 56, 96]", _shape_param_1, _shape_param_2, _shape_param_3, primals_11: "f32[96, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[4, 56, 56, 96]" = torch.ops.aten.reshape.default(mm_101, _shape_param_0);  mm_101 = _shape_param_0 = None
        mul_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(reshape_default, primals_14);  reshape_default = primals_14 = None
        mul_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, 96)
        sum_dim_int_list: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_5);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul_5, sum_dim_int_list_1);  mul_5 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 56, 56, 96]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 56, 56, 96]" = torch.ops.aten.mul.Tensor(div_61, sub_tensor_1);  div_61 = sub_tensor_1 = None
        add_tensor: "f32[4, 56, 56, 96]" = torch.ops.aten.add.Tensor(add_225, mul_tensor_4);  add_225 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:220 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        reshape_default_1: "f32[4, 8, 7, 8, 7, 96]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[4, 8, 8, 7, 7, 96]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:219 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        clone_default: "f32[4, 8, 8, 7, 7, 96]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[256, 49, 96]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        reshape_default_3: "f32[12544, 96]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[96, 96]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_2: "f32[96, 96]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([12544, 96], dtype=torch.float32, device='cuda'),
    [4, 56, 56, 96],  # _shape_param_0
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 56, 56, 96], dtype=torch.float32, device='cuda'),
    [4, 8, 7, 8, 7, 96],  # _shape_param_1
    [256, 49, 96],  # _shape_param_2
    [12544, 96],  # _shape_param_3
    torch.randn([96, 96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
