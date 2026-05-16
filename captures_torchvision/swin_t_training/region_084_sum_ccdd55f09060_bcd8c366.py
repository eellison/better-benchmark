"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: ccdd55f09060
Shape hash: bcd8c366
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_53: "f32[784, 384]", _shape_param_0, _shape_param_1, _shape_param_2, primals_96: "f32[384]", mul_64: "f32[4, 14, 14, 384]", div_48: "f32[4, 14, 14, 1]", add_174: "f32[4, 14, 14, 384]", lt_9: "b8[4, 1, 1, 1]", _shape_param_3, primals_94: "f32[384, 1536]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default: "f32[16, 49, 384]" = torch.ops.aten.reshape.default(mm_53, _shape_param_0);  mm_53 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:172 in shifted_window_attention, code: x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        reshape_default_1: "f32[4, 2, 2, 7, 7, 384]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[4, 2, 7, 2, 7, 384]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2, 4, 5]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:171 in shifted_window_attention, code: x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        clone_default: "f32[4, 2, 7, 2, 7, 384]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[4, 14, 14, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:453 in forward, code: x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        mul_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(reshape_default_2, primals_96);  reshape_default_2 = primals_96 = None
        mul_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, 384)
        sum_dim_int_list: "f32[4, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_64);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_64, sum_dim_int_list_1);  mul_64 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 14, 14, 384]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(div_48, sub_tensor_1);  div_48 = sub_tensor_1 = None
        add_tensor: "f32[4, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_174, mul_tensor_4);  add_174 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_9, torch.float32);  lt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9090909090909091);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor_5: "f32[4, 14, 14, 384]" = torch.ops.aten.mul.Tensor(add_tensor, div_tensor);  add_tensor = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:454 in forward, code: x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        reshape_default_3: "f32[784, 384]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_3);  mul_tensor_5 = _shape_param_3 = None
        permute_default_1: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_default_2: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([784, 384], dtype=torch.float32, device='cuda'),
    [16, 49, 384],  # _shape_param_0
    [4, 2, 2, 7, 7, 384],  # _shape_param_1
    [4, 14, 14, 384],  # _shape_param_2
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1, 1, 1], dtype=torch.bool, device='cuda'),
    [784, 384],  # _shape_param_3
    torch.randn([384, 1536], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
