"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 1176fbb3e33e
Shape hash: 3a7c9345
"""
_shapes_config = "(T([512, 112, 14, 14], f32), T([512, 112, 14, 14], f32, stride=(21952, 1, 1568, 112)), T([1, 112, 1, 1], f32), T([112], f32), T([112], f32), T([512, 480, 1, 1], f32), T([512, 480, 14, 14], f32, stride=(94080, 1, 6720, 480)), T([512, 480, 1, 1], f32), T([512, 240, 14, 14], f32, stride=(47040, 1, 3360, 240)), T([1, 240, 1, 1], f32), T([1, 240, 1, 1], f32), T([240], f32), T([240], f32), T([], f32), S([512, 480, 14, 14]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, clone_7: "f32[512, 112, 14, 14]", convolution_57: "f32[512, 112, 14, 14]", unsqueeze_658: "f32[1, 112, 1, 1]", squeeze_154: "f32[112]", primals_324: "f32[112]", convolution_53: "f32[512, 480, 1, 1]", getitem_280: "f32[512, 480, 14, 14]", getitem_286: "f32[512, 480, 1, 1]", convolution_51: "f32[512, 240, 14, 14]", getitem_95: "f32[1, 240, 1, 1]", rsqrt_47: "f32[1, 240, 1, 1]", primals_296: "f32[240]", primals_297: "f32[240]", full_default: "f32[]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        sum_dim_int_list: "f32[112]" = torch.ops.aten.sum.dim_IntList(clone_7, [0, 2, 3])
        sub_tensor: "f32[512, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_658);  convolution_57 = unsqueeze_658 = None
        mul_tensor: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(clone_7, sub_tensor)
        sum_dim_int_list_1: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[112]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[112]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_tensor_4: "f32[112]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_324);  squeeze_154 = primals_324 = None
        unsqueeze_default_6: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 112, 14, 14]" = torch.ops.aten.sub.Tensor(clone_7, mul_tensor_6);  clone_7 = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 480, 1, 1]" = torch.ops.aten.add.Tensor(convolution_53, 3);  convolution_53 = None
        clamp_min_default: "f32[512, 480, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 480, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 480, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor_8: "f32[512, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_280, div_tensor);  getitem_280 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[512, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_286, _shape_param_0);  getitem_286 = _shape_param_0 = None
        div_scalar: "f32[512, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_default, 196);  expand_default = None
        add_tensor_1: "f32[512, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_8, div_scalar);  mul_tensor_8 = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 240, 14, 14]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 240, 480);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_tensor_3: "f32[512, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_95)
        mul_tensor_9: "f32[512, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_47);  sub_tensor_3 = None
        unsqueeze_default_9: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_296, -1)
        unsqueeze_default_10: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, -1);  unsqueeze_default_9 = None
        mul_tensor_10: "f32[512, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_9, unsqueeze_default_10);  mul_tensor_9 = unsqueeze_default_10 = None
        unsqueeze_default_11: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_297, -1);  primals_297 = None
        unsqueeze_default_12: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, -1);  unsqueeze_default_11 = None
        add_tensor_2: "f32[512, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_10, unsqueeze_default_12);  mul_tensor_10 = unsqueeze_default_12 = None
        relu_default: "f32[512, 240, 14, 14]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar: "b8[512, 240, 14, 14]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[512, 240, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor);  le_scalar = full_default = slice_tensor = None
        squeeze_dims: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        unsqueeze_default_13: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_2: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_4: "f32[512, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_default_15);  convolution_51 = unsqueeze_default_15 = None
        mul_tensor_11: "f32[512, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_4)
        sum_dim_int_list_3: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 2, 3]);  mul_tensor_11 = None
        mul_tensor_12: "f32[240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 9.964923469387754e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_17: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_13: "f32[240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 9.964923469387754e-06);  sum_dim_int_list_3 = None
        squeeze_dims_1: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_tensor_14: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_15: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        unsqueeze_default_19: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_20: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_16: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_296);  squeeze_dims_1 = primals_296 = None
        unsqueeze_default_22: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_16, 0);  mul_tensor_16 = None
        unsqueeze_default_23: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_17: "f32[512, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[512, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_17);  where_self = mul_tensor_17 = None
        sub_tensor_6: "f32[512, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_18: "f32[512, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None
        return (mul_tensor_7, mul_tensor_18)



def make_inputs():
    return [
    torch.randn([512, 112, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn(11239424, dtype=torch.float32, device='cuda').as_strided([512, 112, 14, 14], [21952, 1, 1568, 112]),  # convolution_57
    torch.randn([1, 112, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([512, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(48168960, dtype=torch.float32, device='cuda').as_strided([512, 480, 14, 14], [94080, 1, 6720, 480]),  # getitem_280
    torch.randn([512, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 240, 14, 14], [47040, 1, 3360, 240]),  # convolution_51
    torch.randn([1, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [512, 480, 14, 14],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
