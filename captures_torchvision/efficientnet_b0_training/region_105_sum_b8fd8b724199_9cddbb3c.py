"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: b8fd8b724199
Shape hash: 9cddbb3c
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_161: "f32[4, 672, 7, 7]", sigmoid_11: "f32[4, 672, 1, 1]", getitem_167: "f32[4, 672, 1, 1]", _shape_param_0, add_215: "f32[4, 672, 7, 7]", add_214: "f32[4, 672, 7, 7]", getitem_69: "f32[1, 672, 1, 1]", convolution_56: "f32[4, 672, 7, 7]", rsqrt_34: "f32[1, 672, 1, 1]", primals_254: "f32[672]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_tensor: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_161, sigmoid_11);  getitem_161 = sigmoid_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        expand_default: "f32[4, 672, 7, 7]" = torch.ops.aten.expand.default(getitem_167, _shape_param_0);  getitem_167 = _shape_param_0 = None
        div_scalar: "f32[4, 672, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        add_tensor: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        reciprocal_default: "f32[4, 672, 7, 7]" = torch.ops.aten.reciprocal.default(add_215);  add_215 = None
        mul_tensor_1: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_2: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_1);  add_tensor = None
        sub_tensor: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_3: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_214, sub_tensor);  add_214 = sub_tensor = None
        add_tensor_1: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, 1);  mul_tensor_3 = None
        mul_tensor_4: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1);  mul_tensor_2 = add_tensor_1 = None
        squeeze_dims: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        unsqueeze_default: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sum_dim_int_list: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3])
        sub_tensor_1: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_default_2);  convolution_56 = unsqueeze_default_2 = None
        mul_tensor_5: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sub_tensor_1)
        sum_dim_int_list_1: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 2, 3]);  mul_tensor_5 = None
        mul_tensor_6: "f32[672]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00510204081632653);  sum_dim_int_list = None
        unsqueeze_default_3: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_4: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_7: "f32[672]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00510204081632653);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_tensor_8: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_9: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_8);  mul_tensor_7 = mul_tensor_8 = None
        unsqueeze_default_6: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_7: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_10: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_254);  squeeze_dims_1 = primals_254 = None
        unsqueeze_default_9: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_10: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_11: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_8);  sub_tensor_1 = unsqueeze_default_8 = None
        sub_tensor_2: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(mul_tensor_4, mul_tensor_11);  mul_tensor_4 = mul_tensor_11 = None
        sub_tensor_3: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_5);  sub_tensor_2 = unsqueeze_default_5 = None
        mul_tensor_12: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_11);  sub_tensor_3 = unsqueeze_default_11 = None
        return mul_tensor_12



def make_inputs():
    return [
    torch.randn(131712, dtype=torch.float32, device='cuda').as_strided([4, 672, 7, 7], [32928, 1, 4704, 672]),  # getitem_161
    torch.randn([4, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 672, 1, 1], dtype=torch.float32, device='cuda'),
    [4, 672, 7, 7],  # _shape_param_0
    torch.randn(131712, dtype=torch.float32, device='cuda').as_strided([4, 672, 7, 7], [32928, 1, 4704, 672]),  # add_215
    torch.randn(131712, dtype=torch.float32, device='cuda').as_strided([4, 672, 7, 7], [32928, 1, 4704, 672]),  # add_214
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(131712, dtype=torch.float32, device='cuda').as_strided([4, 672, 7, 7], [32928, 1, 4704, 672]),  # convolution_56
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
