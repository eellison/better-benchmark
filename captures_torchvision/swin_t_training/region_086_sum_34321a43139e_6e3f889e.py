"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 34321a43139e
Shape hash: 6e3f889e
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_25: "f32[784, 384]", _shape_param_0, primals_146: "f32[384]", mul_98: "f32[4, 14, 14, 384]", div_41: "f32[4, 14, 14, 1]", add_145: "f32[4, 14, 14, 384]", lt_16: "b8[4, 1, 1, 1]", _shape_param_1, _shape_param_2, _shape_param_3, primals_143: "f32[384, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[4, 14, 14, 384]" = torch.ops.aten.reshape.default(mm_25, _shape_param_0);  mm_25 = _shape_param_0 = None
        mul_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(reshape_default, primals_146);  reshape_default = primals_146 = None
        mul_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, 384)
        sum_dim_int_list: "f32[4, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_98);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_98, sum_dim_int_list_1);  mul_98 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_41, sub_tensor_1);  div_41 = sub_tensor_1 = None
        add_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_145, mul_tensor_4);  add_145 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_16, torch.float32);  lt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8363636363636364);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor_5: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(add_tensor, div_tensor);  add_tensor = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:167 in shifted_window_attention, code: x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        iota_default: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[14]" = torch.ops.aten.add.Tensor(iota_default, 3);  iota_default = None
        fmod_scalar: "i64[14]" = torch.ops.aten.fmod.Scalar(add_tensor_1, 14);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:224 in shifted_window_attention, code: x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        index_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.index.Tensor(mul_tensor_5, [None, None, fmod_scalar]);  mul_tensor_5 = None
        index_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.index.Tensor(index_tensor, [None, fmod_scalar]);  index_tensor = fmod_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:220 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        reshape_default_1: "f32[4, 2, 7, 2, 7, 384]" = torch.ops.aten.reshape.default(index_tensor_1, _shape_param_1);  index_tensor_1 = _shape_param_1 = None
        permute_default: "f32[4, 2, 2, 7, 7, 384]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:219 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        clone_default: "f32[4, 2, 2, 7, 7, 384]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[16, 49, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        reshape_default_3: "f32[784, 384]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[384, 384]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
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
