"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 80f66f4aea46
Shape hash: 6f6d768c
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_63: "f32[4, 1152, 1, 1]", convolution_61: "f32[4, 1152, 7, 7]", getitem_75: "f32[1, 1152, 1, 1]", rsqrt_37: "f32[1, 1152, 1, 1]", primals_276: "f32[1152]", primals_277: "f32[1152]", getitem_146: "f32[4, 1152, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        sigmoid_default: "f32[4, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        sub_tensor: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_75);  convolution_61 = getitem_75 = None
        mul_tensor: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_37);  sub_tensor = rsqrt_37 = None
        unsqueeze_default: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1);  primals_276 = None
        unsqueeze_default_1: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_default_3: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        neg_default: "f32[4, 1152, 7, 7]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[4, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_tensor_2: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_146, div_tensor);  getitem_146 = div_tensor = None
        sum_dim_int_list: "f32[4, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        sub_tensor_1: "f32[4, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_3: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_4: "f32[4, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_3);  sum_dim_int_list = mul_tensor_3 = None
        return mul_tensor_4



def make_inputs():
    return [
    torch.randn([4, 1152, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(225792, dtype=torch.float32, device='cuda').as_strided([4, 1152, 7, 7], [56448, 1, 8064, 1152]),  # convolution_61
    torch.randn([1, 1152, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1152, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn(225792, dtype=torch.float32, device='cuda').as_strided([4, 1152, 7, 7], [56448, 1, 8064, 1152]),  # getitem_146
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
