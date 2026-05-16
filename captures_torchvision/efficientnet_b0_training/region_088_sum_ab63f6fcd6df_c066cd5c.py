"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: ab63f6fcd6df
Shape hash: c066cd5c
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, lt_1: "b8[4, 1, 1, 1]", getitem_263: "f32[4, 40, 28, 28]", convolution_24: "f32[4, 40, 28, 28]", unsqueeze_606: "f32[1, 40, 1, 1]", squeeze_43: "f32[40]", primals_110: "f32[40]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_1, torch.float32);  lt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.95);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_263, div_tensor);  getitem_263 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        sum_dim_int_list: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3])
        sub_tensor: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_606);  convolution_24 = unsqueeze_606 = None
        mul_tensor_1: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 2, 3]);  mul_tensor_1 = None
        mul_tensor_2: "f32[40]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00031887755102040814);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_2, 0);  mul_tensor_2 = None
        unsqueeze_default_1: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_3: "f32[40]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00031887755102040814);  sum_dim_int_list_1 = None
        mul_tensor_4: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_tensor_5: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        unsqueeze_default_3: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_4: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_6: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_110);  squeeze_43 = primals_110 = None
        unsqueeze_default_6: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_7: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_7: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_7);  mul_tensor = mul_tensor_7 = None
        sub_tensor_2: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_8: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_8



def make_inputs():
    return [
    torch.randint(0, 2, [4, 1, 1, 1], dtype=torch.bool, device='cuda'),
    torch.randn(125440, dtype=torch.float32, device='cuda').as_strided([4, 40, 28, 28], [31360, 1, 1120, 40]),  # getitem_263
    torch.randn(125440, dtype=torch.float32, device='cuda').as_strided([4, 40, 28, 28], [31360, 1, 1120, 40]),  # convolution_24
    torch.randn([1, 40, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
