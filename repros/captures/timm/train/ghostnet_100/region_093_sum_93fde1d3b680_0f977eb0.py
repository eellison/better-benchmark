"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 93fde1d3b680
Shape hash: 0f977eb0
"""
_shapes_config = "(T([512, 200, 14, 14], f32, stride=(39200, 1, 2800, 200)), T([512, 100, 14, 14], f32, stride=(19600, 1, 1400, 100)), T([512, 100, 14, 14], f32, stride=(19600, 1, 1400, 100)), T([], f32), T([512, 100, 14, 14], f32, stride=(19600, 1, 1400, 100)), T([1, 100, 1, 1], f32), T([100], f32), T([100], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_322: "f32[512, 200, 14, 14]", getitem_325: "f32[512, 100, 14, 14]", relu_15: "f32[512, 100, 14, 14]", full_default: "f32[]", convolution_38: "f32[512, 100, 14, 14]", unsqueeze_862: "f32[1, 100, 1, 1]", squeeze_103: "f32[100]", primals_218: "f32[100]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 100, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_322, 1, 0, 100);  getitem_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor: "f32[512, 100, 14, 14]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_325);  slice_tensor = getitem_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_scalar: "b8[512, 100, 14, 14]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_self: "f32[512, 100, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None
        sum_dim_int_list: "f32[100]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[512, 100, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_862);  convolution_38 = unsqueeze_862 = None
        mul_tensor: "f32[512, 100, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[100]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[100]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[100]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_tensor_4: "f32[100]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_218);  squeeze_103 = primals_218 = None
        unsqueeze_default_6: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 100, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 100, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(20070400, dtype=torch.float32, device='cuda').as_strided([512, 200, 14, 14], [39200, 1, 2800, 200]),  # getitem_322
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([512, 100, 14, 14], [19600, 1, 1400, 100]),  # getitem_325
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([512, 100, 14, 14], [19600, 1, 1400, 100]),  # relu_15
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([512, 100, 14, 14], [19600, 1, 1400, 100]),  # convolution_38
    torch.randn([1, 100, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([100], dtype=torch.float32, device='cuda'),
    torch.randn([100], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
