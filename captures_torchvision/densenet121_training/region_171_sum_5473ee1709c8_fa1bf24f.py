"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: 5473ee1709c8
Shape hash: fa1bf24f
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mul_1151: "f32[4, 1024, 14, 14]", mul_1169: "f32[4, 992, 14, 14]", mul_1187: "f32[4, 960, 14, 14]", mul_1205: "f32[4, 928, 14, 14]", mul_1223: "f32[4, 896, 14, 14]", mul_1241: "f32[4, 864, 14, 14]", mul_1259: "f32[4, 832, 14, 14]", mul_1277: "f32[4, 800, 14, 14]", relu_71: "f32[4, 768, 14, 14]", full_default: "f32[]", getitem_388: "f32[4, 768, 14, 14]", cat_33: "f32[4, 768, 14, 14]", unsqueeze_1074: "f32[1, 768, 1, 1]", squeeze_214: "f32[768]", primals_431: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_tensor: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1151, 1, 736, 768);  mul_1151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_1: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1169, 1, 736, 768);  mul_1169 = None
        add_tensor: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1187, 1, 736, 768);  mul_1187 = None
        add_tensor_1: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1205, 1, 736, 768);  mul_1205 = None
        add_tensor_2: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1223, 1, 736, 768);  mul_1223 = None
        add_tensor_3: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1241, 1, 736, 768);  mul_1241 = None
        add_tensor_4: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        slice_tensor_6: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1259, 1, 736, 768);  mul_1259 = None
        add_tensor_5: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        slice_tensor_7: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1277, 1, 736, 768);  mul_1277 = None
        add_tensor_6: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        le_scalar: "b8[4, 768, 14, 14]" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
        where_self: "f32[4, 768, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_388);  le_scalar = full_default = getitem_388 = None
        sum_dim_int_list: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[4, 768, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, unsqueeze_1074);  cat_33 = unsqueeze_1074 = None
        mul_tensor: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0012755102040816326);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0012755102040816326);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
        mul_tensor_4: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_431);  squeeze_214 = primals_431 = None
        unsqueeze_default_6: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 768, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 768, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_8: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 736, 768);  mul_tensor_7 = None
        add_tensor_7: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None
        return add_tensor_7



def make_inputs():
    return [
    torch.randn([4, 1024, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 992, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 960, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 928, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 896, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 864, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 832, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 800, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 768, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([4, 768, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 768, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
