"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: 522a725c3033
Shape hash: 9a251acb
"""
_shapes_config = "(T([128, 672, 14, 14], f32, stride=(131712, 1, 9408, 672)), T([128, 672, 1, 1], f32), T([128, 672, 1, 1], f32), T([128, 672, 14, 14], f32, stride=(131712, 1, 9408, 672)), T([128, 672, 14, 14], f32, stride=(131712, 1, 9408, 672)), T([1, 672, 1, 1], f32), T([128, 672, 14, 14], f32, stride=(131712, 1, 9408, 672)), T([1, 672, 1, 1], f32), T([672], f32), S([128, 672, 14, 14]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_191: "f32[128, 672, 14, 14]", sigmoid_9: "f32[128, 672, 1, 1]", getitem_197: "f32[128, 672, 1, 1]", add_177: "f32[128, 672, 14, 14]", add_176: "f32[128, 672, 14, 14]", getitem_57: "f32[1, 672, 1, 1]", convolution_46: "f32[128, 672, 14, 14]", rsqrt_28: "f32[1, 672, 1, 1]", primals_210: "f32[672]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_191, sigmoid_9);  getitem_191 = sigmoid_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[128, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_197, _shape_param_0);  getitem_197 = _shape_param_0 = None
        div_scalar: "f32[128, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_default, 196);  expand_default = None
        add_tensor: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_default: "f32[128, 672, 14, 14]" = torch.ops.aten.reciprocal.default(add_177);  add_177 = None
        mul_tensor_1: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_2: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_1);  add_tensor = None
        sub_tensor: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_3: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_176, sub_tensor);  add_176 = sub_tensor = None
        add_tensor_1: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_3, 1);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1);  mul_tensor_2 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        unsqueeze_default: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sum_dim_int_list: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3])
        sub_tensor_1: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_default_2);  convolution_46 = unsqueeze_default_2 = None
        mul_tensor_5: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sub_tensor_1)
        sum_dim_int_list_1: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 2, 3]);  mul_tensor_5 = None
        mul_tensor_6: "f32[672]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default_3: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_4: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_7: "f32[672]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_tensor_8: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_9: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_8);  mul_tensor_7 = mul_tensor_8 = None
        unsqueeze_default_6: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_7: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_10: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_210);  squeeze_dims_1 = primals_210 = None
        unsqueeze_default_9: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_10: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_11: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_8);  sub_tensor_1 = unsqueeze_default_8 = None
        sub_tensor_2: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_tensor_4, mul_tensor_11);  mul_tensor_4 = mul_tensor_11 = None
        sub_tensor_3: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_5);  sub_tensor_2 = unsqueeze_default_5 = None
        mul_tensor_12: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_11);  sub_tensor_3 = unsqueeze_default_11 = None
        return mul_tensor_12



def make_inputs():
    return [
    torch.randn(16859136, dtype=torch.float32, device='cuda').as_strided([128, 672, 14, 14], [131712, 1, 9408, 672]),  # getitem_191
    torch.randn([128, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(16859136, dtype=torch.float32, device='cuda').as_strided([128, 672, 14, 14], [131712, 1, 9408, 672]),  # add_177
    torch.randn(16859136, dtype=torch.float32, device='cuda').as_strided([128, 672, 14, 14], [131712, 1, 9408, 672]),  # add_176
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(16859136, dtype=torch.float32, device='cuda').as_strided([128, 672, 14, 14], [131712, 1, 9408, 672]),  # convolution_46
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    [128, 672, 14, 14],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
