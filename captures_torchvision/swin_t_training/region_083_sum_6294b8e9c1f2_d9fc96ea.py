"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 6294b8e9c1f2
Shape hash: d9fc96ea
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_57: "f32[784, 384]", _shape_param_0, primals_90: "f32[384]", mul_58: "f32[4, 14, 14, 384]", div_49: "f32[4, 14, 14, 1]", add_177: "f32[4, 14, 14, 384]", lt_8: "b8[4, 1, 1, 1]", fmod_8: "i64[14]", _shape_param_1, _shape_param_2, _shape_param_3, primals_87: "f32[384, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[4, 14, 14, 384]" = torch.ops.aten.reshape.default(mm_57, _shape_param_0);  mm_57 = _shape_param_0 = None
        mul_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(reshape_default, primals_90);  reshape_default = primals_90 = None
        mul_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, 384)
        sum_dim_int_list: "f32[4, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_58);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_58, sum_dim_int_list_1);  mul_58 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_49, sub_tensor_1);  div_49 = sub_tensor_1 = None
        add_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_177, mul_tensor_4);  add_177 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_8, torch.float32);  lt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9090909090909091);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor_5: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(add_tensor, div_tensor);  add_tensor = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:224 in shifted_window_attention, code: x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        index_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.index.Tensor(mul_tensor_5, [None, None, fmod_8]);  mul_tensor_5 = None
        index_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.index.Tensor(index_tensor, [None, fmod_8]);  index_tensor = fmod_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:220 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        reshape_default_1: "f32[4, 2, 7, 2, 7, 384]" = torch.ops.aten.reshape.default(index_tensor_1, _shape_param_1);  index_tensor_1 = _shape_param_1 = None
        permute_default: "f32[4, 2, 2, 7, 7, 384]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:219 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        clone_default: "f32[4, 2, 2, 7, 7, 384]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[16, 49, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        reshape_default_3: "f32[784, 384]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[384, 384]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_default_2: "f32[384, 384]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [4, 14, 14, 384],  # _shape_param_0
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1, 1, 1], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [14], dtype=torch.int64, device='cuda'),
    [4, 2, 7, 2, 7, 384],  # _shape_param_1
    [16, 49, 384],  # _shape_param_2
    [784, 384],  # _shape_param_3
    torch.randn([384, 384], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
