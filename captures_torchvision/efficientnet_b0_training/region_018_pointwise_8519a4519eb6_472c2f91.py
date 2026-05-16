"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 8519a4519eb6
Shape hash: 472c2f91
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_64: "f32[1, 112, 1, 1]", convolution_54: "f32[4, 112, 14, 14]", getitem_65: "f32[1, 112, 1, 1]", primals_242: "f32[112]", primals_243: "f32[112]", inductor_seeds_default: "i64[10]", add_184: "f32[4, 112, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        add_tensor: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_default: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_65);  convolution_54 = getitem_65 = None
        mul_tensor: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1);  primals_242 = None
        unsqueeze_default_1: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_243, -1);  primals_243 = None
        unsqueeze_default_3: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 1, 1, 1]" = torch.ops.prims.inductor_random.default([4, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[4, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.875);  inductor_random_default = None
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.875);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor_2: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor_1, div_tensor);  add_tensor_1 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:168 in forward, code: result += input
        add_tensor_2: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, add_184);  mul_tensor_2 = add_184 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn([1, 112, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 112, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([1, 112, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [10], dtype=torch.int64, device='cuda'),
    torch.randn([4, 112, 14, 14], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
