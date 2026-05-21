"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: f9cf0f90e7dc
Shape hash: 89754c99
"""
_shapes_config = "(T([128, 1280, 8, 8], f32, stride=(81920, 1, 10240, 1280)), T([128, 1280, 8, 8], f32, stride=(81920, 1, 10240, 1280)), T([128, 1280, 8, 8], f32, stride=(81920, 1, 10240, 1280)), T([128, 1280, 8, 8], f32, stride=(81920, 1, 10240, 1280)), T([128, 1280, 8, 8], f32, stride=(81920, 1, 10240, 1280)), T([128, 192, 8, 8], f32, stride=(12288, 1, 1536, 192)), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_223: "f32[128, 1280, 8, 8]", cat_8: "f32[128, 1280, 8, 8]", getitem_235: "f32[128, 1280, 8, 8]", getitem_244: "f32[128, 1280, 8, 8]", getitem_247: "f32[128, 1280, 8, 8]", convolution_75: "f32[128, 192, 8, 8]", getitem_157: "f32[1, 192, 1, 1]", rsqrt_75: "f32[1, 192, 1, 1]", primals_456: "f32[192]", primals_457: "f32[192]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:236 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_default: "f32[128, 1280, 8, 8]" = torch.ops.aten.avg_pool2d_backward.default(getitem_223, cat_8, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_223 = cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[128, 1280, 8, 8]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_default, getitem_235);  avg_pool2d_backward_default = getitem_235 = None
        add_tensor_1: "f32[128, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_tensor, getitem_244);  add_tensor = getitem_244 = None
        add_tensor_2: "f32[128, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_tensor_1, getitem_247);  add_tensor_1 = getitem_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:190 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 192, 8, 8]" = torch.ops.aten.slice.Tensor(add_tensor_2, 1, 320, 512);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_157)
        mul_tensor: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_75);  sub_tensor = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_456, -1)
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_457, -1);  primals_457 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 192, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        le_scalar: "b8[128, 192, 8, 8]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 192, 8, 8]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor);  le_scalar = full_default = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        unsqueeze_default_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_default_6);  convolution_75 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0001220703125);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0001220703125);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_456);  squeeze_dims_1 = primals_456 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_9



def make_inputs():
    return [
    torch.randn(10485760, dtype=torch.float32, device='cuda').as_strided([128, 1280, 8, 8], [81920, 1, 10240, 1280]),  # getitem_223
    torch.randn(10485760, dtype=torch.float32, device='cuda').as_strided([128, 1280, 8, 8], [81920, 1, 10240, 1280]),  # cat_8
    torch.randn(10485760, dtype=torch.float32, device='cuda').as_strided([128, 1280, 8, 8], [81920, 1, 10240, 1280]),  # getitem_235
    torch.randn(10485760, dtype=torch.float32, device='cuda').as_strided([128, 1280, 8, 8], [81920, 1, 10240, 1280]),  # getitem_244
    torch.randn(10485760, dtype=torch.float32, device='cuda').as_strided([128, 1280, 8, 8], [81920, 1, 10240, 1280]),  # getitem_247
    torch.randn(1572864, dtype=torch.float32, device='cuda').as_strided([128, 192, 8, 8], [12288, 1, 1536, 192]),  # convolution_75
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
