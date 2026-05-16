"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: ac3adbacf057
Shape hash: 5e100560
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mul_854: "f32[4, 1024, 7, 7]", mul_872: "f32[4, 992, 7, 7]", mul_890: "f32[4, 960, 7, 7]", mul_908: "f32[4, 928, 7, 7]", mul_926: "f32[4, 896, 7, 7]", mul_944: "f32[4, 864, 7, 7]", mul_962: "f32[4, 832, 7, 7]", mul_980: "f32[4, 800, 7, 7]", mul_998: "f32[4, 768, 7, 7]", mul_1016: "f32[4, 736, 7, 7]", mul_1034: "f32[4, 704, 7, 7]", mul_1052: "f32[4, 672, 7, 7]", relu_96: "f32[4, 640, 7, 7]", full_default: "f32[]", getitem_313: "f32[4, 640, 7, 7]", cat_45: "f32[4, 640, 7, 7]", unsqueeze_774: "f32[1, 640, 1, 1]", squeeze_289: "f32[640]", primals_581: "f32[640]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_tensor: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_854, 1, 608, 640);  mul_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_1: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_872, 1, 608, 640);  mul_872 = None
        add_tensor: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_890, 1, 608, 640);  mul_890 = None
        add_tensor_1: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_908, 1, 608, 640);  mul_908 = None
        add_tensor_2: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_926, 1, 608, 640);  mul_926 = None
        add_tensor_3: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_944, 1, 608, 640);  mul_944 = None
        add_tensor_4: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        slice_tensor_6: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_962, 1, 608, 640);  mul_962 = None
        add_tensor_5: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        slice_tensor_7: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_980, 1, 608, 640);  mul_980 = None
        add_tensor_6: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None
        slice_tensor_8: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_998, 1, 608, 640);  mul_998 = None
        add_tensor_7: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None
        slice_tensor_9: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_1016, 1, 608, 640);  mul_1016 = None
        add_tensor_8: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_7, slice_tensor_9);  add_tensor_7 = slice_tensor_9 = None
        slice_tensor_10: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_1034, 1, 608, 640);  mul_1034 = None
        add_tensor_9: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_8, slice_tensor_10);  add_tensor_8 = slice_tensor_10 = None
        slice_tensor_11: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_1052, 1, 608, 640);  mul_1052 = None
        add_tensor_10: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_9, slice_tensor_11);  add_tensor_9 = slice_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        le_scalar: "b8[4, 640, 7, 7]" = torch.ops.aten.le.Scalar(relu_96, 0);  relu_96 = None
        where_self: "f32[4, 640, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_313);  le_scalar = full_default = getitem_313 = None
        sum_dim_int_list: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[4, 640, 7, 7]" = torch.ops.aten.sub.Tensor(cat_45, unsqueeze_774);  cat_45 = unsqueeze_774 = None
        mul_tensor: "f32[4, 640, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[640]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00510204081632653);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[640]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00510204081632653);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
        mul_tensor_4: "f32[640]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_581);  squeeze_289 = primals_581 = None
        unsqueeze_default_6: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 640, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 640, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 640, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 640, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_12: "f32[4, 32, 7, 7]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 608, 640);  mul_tensor_7 = None
        add_tensor_11: "f32[4, 32, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_10, slice_tensor_12);  add_tensor_10 = slice_tensor_12 = None
        return add_tensor_11



def make_inputs():
    return [
    torch.randn([4, 1024, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 992, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 960, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 928, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 896, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 864, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 832, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 800, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 768, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 736, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 704, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 672, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 640, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([4, 640, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([4, 640, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 640, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
