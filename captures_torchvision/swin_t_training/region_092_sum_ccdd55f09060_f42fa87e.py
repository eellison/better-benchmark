"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: ccdd55f09060
Shape hash: f42fa87e
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_15: "f32[196, 768]", _shape_param_0, primals_163: "f32[768]", mul_110: "f32[4, 7, 7, 768]", div_38: "f32[4, 7, 7, 1]", add_136: "f32[4, 7, 7, 768]", lt_18: "b8[4, 1, 1, 1]", _shape_param_1, _shape_param_2, _shape_param_3, primals_160: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default: "f32[4, 7, 7, 768]" = torch.ops.aten.reshape.default(mm_15, _shape_param_0);  mm_15 = _shape_param_0 = None
        mul_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_163);  reshape_default = primals_163 = None
        mul_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_110);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_110, sum_dim_int_list_1);  mul_110 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_tensor_1);  div_38 = sub_tensor_1 = None
        add_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.add.Tensor(add_136, mul_tensor_4);  add_136 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_18, torch.float32);  lt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8181818181818181);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor_5: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(add_tensor, div_tensor);  add_tensor = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:220 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
        reshape_default_1: "f32[4, 1, 7, 1, 7, 768]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_1);  mul_tensor_5 = _shape_param_1 = None
        permute_default: "f32[4, 1, 1, 7, 7, 768]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:219 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_2: "f32[4, 49, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        clone_default: "f32[4, 49, 768]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.contiguous_format);  reshape_default_2 = None
        reshape_default_3: "f32[196, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    [4, 7, 7, 768],  # _shape_param_0
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn(150528, dtype=torch.float32, device='cuda').as_strided([4, 7, 7, 768], [37632, 7, 1, 49]),  # add_136
    torch.randint(0, 2, [4, 1, 1, 1], dtype=torch.bool, device='cuda'),
    [4, 1, 7, 1, 7, 768],  # _shape_param_1
    [4, 49, 768],  # _shape_param_2
    [196, 768],  # _shape_param_3
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
