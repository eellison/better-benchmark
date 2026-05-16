"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: 0ebef8c89a6d
Shape hash: 3dfcfc29
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mul_1151: "f32[4, 1024, 14, 14]", mul_1169: "f32[4, 992, 14, 14]", mul_1187: "f32[4, 960, 14, 14]", mul_1205: "f32[4, 928, 14, 14]", mul_1223: "f32[4, 896, 14, 14]", mul_1241: "f32[4, 864, 14, 14]", mul_1259: "f32[4, 832, 14, 14]", mul_1277: "f32[4, 800, 14, 14]", mul_1295: "f32[4, 768, 14, 14]", mul_1313: "f32[4, 736, 14, 14]", mul_1331: "f32[4, 704, 14, 14]", mul_1349: "f32[4, 672, 14, 14]", mul_1367: "f32[4, 640, 14, 14]", mul_1385: "f32[4, 608, 14, 14]", mul_1403: "f32[4, 576, 14, 14]", mul_1421: "f32[4, 544, 14, 14]", mul_1439: "f32[4, 512, 14, 14]", mul_1457: "f32[4, 480, 14, 14]", mul_1475: "f32[4, 448, 14, 14]", mul_1493: "f32[4, 416, 14, 14]", mul_1511: "f32[4, 384, 14, 14]", mul_1529: "f32[4, 352, 14, 14]", mul_1547: "f32[4, 320, 14, 14]", relu_41: "f32[4, 288, 14, 14]", full_default: "f32[]", getitem_478: "f32[4, 288, 14, 14]", cat_18: "f32[4, 288, 14, 14]", unsqueeze_1434: "f32[1, 288, 1, 1]", squeeze_124: "f32[288]", primals_251: "f32[288]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_tensor: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1151, 1, 256, 288);  mul_1151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_1: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1169, 1, 256, 288);  mul_1169 = None
        add_tensor: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1187, 1, 256, 288);  mul_1187 = None
        add_tensor_1: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1205, 1, 256, 288);  mul_1205 = None
        add_tensor_2: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1223, 1, 256, 288);  mul_1223 = None
        add_tensor_3: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1241, 1, 256, 288);  mul_1241 = None
        add_tensor_4: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        slice_tensor_6: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1259, 1, 256, 288);  mul_1259 = None
        add_tensor_5: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_4, slice_tensor_6);  add_tensor_4 = slice_tensor_6 = None
        slice_tensor_7: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1277, 1, 256, 288);  mul_1277 = None
        add_tensor_6: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_5, slice_tensor_7);  add_tensor_5 = slice_tensor_7 = None
        slice_tensor_8: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1295, 1, 256, 288);  mul_1295 = None
        add_tensor_7: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_6, slice_tensor_8);  add_tensor_6 = slice_tensor_8 = None
        slice_tensor_9: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1313, 1, 256, 288);  mul_1313 = None
        add_tensor_8: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_7, slice_tensor_9);  add_tensor_7 = slice_tensor_9 = None
        slice_tensor_10: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1331, 1, 256, 288);  mul_1331 = None
        add_tensor_9: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_8, slice_tensor_10);  add_tensor_8 = slice_tensor_10 = None
        slice_tensor_11: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1349, 1, 256, 288);  mul_1349 = None
        add_tensor_10: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_9, slice_tensor_11);  add_tensor_9 = slice_tensor_11 = None
        slice_tensor_12: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1367, 1, 256, 288);  mul_1367 = None
        add_tensor_11: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_10, slice_tensor_12);  add_tensor_10 = slice_tensor_12 = None
        slice_tensor_13: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1385, 1, 256, 288);  mul_1385 = None
        add_tensor_12: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_11, slice_tensor_13);  add_tensor_11 = slice_tensor_13 = None
        slice_tensor_14: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1403, 1, 256, 288);  mul_1403 = None
        add_tensor_13: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_12, slice_tensor_14);  add_tensor_12 = slice_tensor_14 = None
        slice_tensor_15: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1421, 1, 256, 288);  mul_1421 = None
        add_tensor_14: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_13, slice_tensor_15);  add_tensor_13 = slice_tensor_15 = None
        slice_tensor_16: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1439, 1, 256, 288);  mul_1439 = None
        add_tensor_15: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_14, slice_tensor_16);  add_tensor_14 = slice_tensor_16 = None
        slice_tensor_17: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1457, 1, 256, 288);  mul_1457 = None
        add_tensor_16: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_15, slice_tensor_17);  add_tensor_15 = slice_tensor_17 = None
        slice_tensor_18: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1475, 1, 256, 288);  mul_1475 = None
        add_tensor_17: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_16, slice_tensor_18);  add_tensor_16 = slice_tensor_18 = None
        slice_tensor_19: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1493, 1, 256, 288);  mul_1493 = None
        add_tensor_18: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_17, slice_tensor_19);  add_tensor_17 = slice_tensor_19 = None
        slice_tensor_20: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1511, 1, 256, 288);  mul_1511 = None
        add_tensor_19: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_18, slice_tensor_20);  add_tensor_18 = slice_tensor_20 = None
        slice_tensor_21: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1529, 1, 256, 288);  mul_1529 = None
        add_tensor_20: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_19, slice_tensor_21);  add_tensor_19 = slice_tensor_21 = None
        slice_tensor_22: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1547, 1, 256, 288);  mul_1547 = None
        add_tensor_21: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_20, slice_tensor_22);  add_tensor_20 = slice_tensor_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        le_scalar: "b8[4, 288, 14, 14]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_self: "f32[4, 288, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_478);  le_scalar = full_default = getitem_478 = None
        sum_dim_int_list: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[4, 288, 14, 14]" = torch.ops.aten.sub.Tensor(cat_18, unsqueeze_1434);  cat_18 = unsqueeze_1434 = None
        mul_tensor: "f32[4, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[288]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0012755102040816326);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[288]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0012755102040816326);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_tensor_4: "f32[288]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_251);  squeeze_124 = primals_251 = None
        unsqueeze_default_6: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[4, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 288, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[4, 288, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[4, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_23: "f32[4, 32, 14, 14]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 256, 288);  mul_tensor_7 = None
        add_tensor_22: "f32[4, 32, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_21, slice_tensor_23);  add_tensor_21 = slice_tensor_23 = None
        return add_tensor_22



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
    torch.randn([4, 736, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 704, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 672, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 640, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 608, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 576, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 544, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 480, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 448, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 416, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 384, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 352, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 320, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 288, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([4, 288, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([4, 288, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([1, 288, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
