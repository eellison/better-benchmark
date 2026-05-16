"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: 10084f604c2b
Shape hash: d29d0cd9
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mul_1592: "f32[4, 512, 28, 28]", mul_1610: "f32[4, 480, 28, 28]", mul_1628: "f32[4, 448, 28, 28]", mul_1646: "f32[4, 416, 28, 28]", mul_1664: "f32[4, 384, 28, 28]", mul_1682: "f32[4, 352, 28, 28]", mul_1700: "f32[4, 320, 28, 28]", mul_1718: "f32[4, 288, 28, 28]", mul_1736: "f32[4, 256, 28, 28]", relu_20: "f32[4, 224, 28, 28]", full_default: "f32[]", getitem_541: "f32[4, 224, 28, 28]", cat_8: "f32[4, 224, 28, 28]", unsqueeze_1686: "f32[1, 224, 1, 1]", squeeze_61: "f32[224]", primals_125: "f32[224]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_tensor: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1592, 1, 192, 224);  mul_1592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_1: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1610, 1, 192, 224);  mul_1610 = None
        add_tensor: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1628, 1, 192, 224);  mul_1628 = None
        add_tensor_1: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1646, 1, 192, 224);  mul_1646 = None
        add_tensor_2: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1664, 1, 192, 224);  mul_1664 = None
        add_tensor_3: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1682, 1, 192, 224);  mul_1682 = None
        add_tensor_4: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        slice_tensor_6: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1700, 1, 192, 224);  mul_1700 = None
        add_tensor_5: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        slice_tensor_7: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1718, 1, 192, 224);  mul_1718 = None
        add_tensor_6: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None
        slice_tensor_8: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1736, 1, 192, 224);  mul_1736 = None
        add_tensor_7: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        le_scalar: "b8[4, 224, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_self: "f32[4, 224, 28, 28]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_541);  le_scalar = full_default = getitem_541 = None
        sum_dim_int_list: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[4, 224, 28, 28]" = torch.ops.aten.sub.Tensor(cat_8, unsqueeze_1686);  cat_8 = unsqueeze_1686 = None
        mul_tensor: "f32[4, 224, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[224]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00031887755102040814);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[224]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00031887755102040814);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_tensor_4: "f32[224]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_125);  squeeze_61 = primals_125 = None
        unsqueeze_default_6: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 224, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 224, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_9: "f32[4, 32, 28, 28]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 192, 224);  mul_tensor_7 = None
        add_tensor_8: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_7, slice_tensor_9);  add_tensor_7 = slice_tensor_9 = None
        return add_tensor_8



def make_inputs():
    return [
    torch.randn([4, 512, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 480, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 448, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 416, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 384, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 352, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 320, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 288, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 256, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 224, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([4, 224, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([4, 224, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([1, 224, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
