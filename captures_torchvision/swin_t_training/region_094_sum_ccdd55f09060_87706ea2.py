"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: ccdd55f09060
Shape hash: 87706ea2
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_11: "f32[196, 768]", _shape_param_0, _shape_param_1, _shape_param_2, primals_169: "f32[768]", mul_116: "f32[4, 7, 7, 768]", div_37: "f32[4, 7, 7, 1]", add_133: "f32[4, 7, 7, 768]", lt_19: "b8[4, 1, 1, 1]", _shape_param_3, primals_167: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default: "f32[4, 49, 768]" = torch.ops.aten.reshape.default(mm_11, _shape_param_0);  mm_11 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:172 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        reshape_default_1: "f32[4, 1, 1, 7, 7, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[4, 1, 7, 1, 7, 768]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:171 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        reshape_default_2: "f32[4, 7, 7, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(reshape_default_2, primals_169);  reshape_default_2 = primals_169 = None
        mul_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_116);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(mul_116, sum_dim_int_list_1);  mul_116 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 7, 7, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_tensor_1);  div_37 = sub_tensor_1 = None
        add_tensor: "f32[4, 7, 7, 768]" = torch.ops.aten.add.Tensor(add_133, mul_tensor_4);  add_133 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_19, torch.float32);  lt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8181818181818181);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor_5: "f32[4, 7, 7, 768]" = torch.ops.aten.mul.Tensor(add_tensor, div_tensor);  add_tensor = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        clone_default: "f32[4, 7, 7, 768]" = torch.ops.aten.clone.default(mul_tensor_5, memory_format = torch.contiguous_format);  mul_tensor_5 = None
        reshape_default_3: "f32[196, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        permute_default_1: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
        permute_default_2: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([196, 768], dtype=torch.float32, device='cuda'),
    [4, 49, 768],  # _shape_param_0
    [4, 1, 1, 7, 7, 768],  # _shape_param_1
    [4, 7, 7, 768],  # _shape_param_2
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 7, 7, 1], dtype=torch.float32, device='cuda'),
    torch.randn(150528, dtype=torch.float32, device='cuda').as_strided([4, 7, 7, 768], [37632, 7, 1, 49]),  # add_133
    torch.randint(0, 2, [4, 1, 1, 1], dtype=torch.bool, device='cuda'),
    [196, 768],  # _shape_param_3
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
