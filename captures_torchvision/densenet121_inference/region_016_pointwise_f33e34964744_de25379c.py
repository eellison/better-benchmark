"""
Standalone repro captured via capture_hook.
Label: densenet121_inference
Pattern hash: f33e34964744
Shape hash: de25379c
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, avg_pool2d_2: "f32[1, 512, 7, 7]", convolution_89: "f32[1, 32, 7, 7]", arg451_1: "f32[544]", arg452_1: "f32[544]", arg453_1: "f32[544]", arg454_1: "f32[544]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_default: "f32[1, 544, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89], 1);  avg_pool2d_2 = convolution_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_default: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(arg451_1, -1);  arg451_1 = None
        unsqueeze_default_1: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[1, 544, 7, 7]" = torch.ops.aten.sub.Tensor(cat_default, unsqueeze_default_1);  cat_default = unsqueeze_default_1 = None
        add_tensor: "f32[544]" = torch.ops.aten.add.Tensor(arg452_1, 1e-05);  arg452_1 = None
        sqrt_default: "f32[544]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[544]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[544]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[1, 544, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(arg453_1, -1);  arg453_1 = None
        unsqueeze_default_5: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[1, 544, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_default_7: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[1, 544, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        relu_default: "f32[1, 544, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([1, 512, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
