"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 2813c3cf2b06
Shape hash: b0f23f1f
"""
_shapes_config = "(T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], f32, stride=(235200, 1, 6720, 192)), T([128, 192, 35, 35], i8, stride=(235200, 1, 6720, 192)), T([128, 192, 71, 71], f32, stride=(967872, 1, 13632, 192)), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([], f32), S([24576, 1225]), S([24576, 1225]), S([128, 192, 71, 71]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_442: "f32[128, 192, 35, 35]", getitem_12: "f32[128, 192, 35, 35]", getitem_451: "f32[128, 192, 35, 35]", getitem_457: "f32[128, 192, 35, 35]", getitem_460: "f32[128, 192, 35, 35]", getitem_13: "i8[128, 192, 35, 35]", convolution_4: "f32[128, 192, 71, 71]", getitem_11: "f32[1, 192, 1, 1]", rsqrt_4: "f32[1, 192, 1, 1]", primals_30: "f32[192]", primals_31: "f32[192]", full_default: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_default: "f32[128, 192, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(getitem_442, getitem_12, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_442 = getitem_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        add_tensor: "f32[128, 192, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_default, getitem_451);  avg_pool2d_backward_default = getitem_451 = None
        add_tensor_1: "f32[128, 192, 35, 35]" = torch.ops.aten.add.Tensor(add_tensor, getitem_457);  add_tensor = getitem_457 = None
        add_tensor_2: "f32[128, 192, 35, 35]" = torch.ops.aten.add.Tensor(add_tensor_1, getitem_460);  add_tensor_1 = getitem_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:399 in forward_preaux, code: x = self.Pool2(x)  # N x 192 x 35 x 35
        full_default_1: "f32[24576, 5041]" = torch.ops.aten.full.default([24576, 5041], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clone_default: "f32[128, 192, 35, 35]" = torch.ops.aten.clone.default(add_tensor_2, memory_format = torch.contiguous_format);  add_tensor_2 = None
        reshape_default: "f32[24576, 1225]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[128, 192, 35, 35]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_13, [3, 3], [71, 71], [2, 2], [0, 0], [1, 1]);  getitem_13 = None
        clone_default_1: "i64[128, 192, 35, 35]" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices_default, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices_default = None
        reshape_default_1: "i64[24576, 1225]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        scatter_add_default: "f32[24576, 5041]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[128, 192, 71, 71]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        clone_default_2: "f32[128, 192, 71, 71]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.channels_last);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_tensor: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_4);  sub_tensor = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 192, 71, 71]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 192, 71, 71]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        le_scalar: "b8[128, 192, 71, 71]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 192, 71, 71]" = torch.ops.aten.where.self(le_scalar, full_default, clone_default_2);  le_scalar = full_default = clone_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        unsqueeze_default_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_default_6);  convolution_4 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 1.5497917079944455e-06);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.5497917079944455e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_30);  squeeze_dims_1 = primals_30 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_9



def make_inputs():
    return [
    torch.randn(30105600, dtype=torch.float32, device='cuda').as_strided([128, 192, 35, 35], [235200, 1, 6720, 192]),  # getitem_442
    torch.randn(30105600, dtype=torch.float32, device='cuda').as_strided([128, 192, 35, 35], [235200, 1, 6720, 192]),  # getitem_12
    torch.randn(30105600, dtype=torch.float32, device='cuda').as_strided([128, 192, 35, 35], [235200, 1, 6720, 192]),  # getitem_451
    torch.randn(30105600, dtype=torch.float32, device='cuda').as_strided([128, 192, 35, 35], [235200, 1, 6720, 192]),  # getitem_457
    torch.randn(30105600, dtype=torch.float32, device='cuda').as_strided([128, 192, 35, 35], [235200, 1, 6720, 192]),  # getitem_460
    torch.randint(0, 9, (30105600,), dtype=torch.int8, device='cuda').as_strided([128, 192, 35, 35], [235200, 1, 6720, 192]),  # getitem_13
    torch.randn(123887616, dtype=torch.float32, device='cuda').as_strided([128, 192, 71, 71], [967872, 1, 13632, 192]),  # convolution_4
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [24576, 1225],  # _shape_param_0
    [24576, 1225],  # _shape_param_1
    [128, 192, 71, 71],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
