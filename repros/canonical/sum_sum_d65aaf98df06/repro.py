"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train
Pattern hash: d65aaf98df06
Shape hash: ba20fbec
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mul_1817: "f32[64, 256, 56, 56]", mul_1835: "f32[64, 224, 56, 56]", mul_1853: "f32[64, 192, 56, 56]", mul_1871: "f32[64, 160, 56, 56]", relu_5: "f32[64, 128, 56, 56]", full_default: "f32[]", getitem_586: "f32[64, 128, 56, 56]", cat_1: "f32[64, 128, 56, 56]", unsqueeze_1866: "f32[1, 128, 1, 1]", squeeze_16: "f32[128]", primals_35: "f32[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_tensor: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1817, 1, 96, 128);  mul_1817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_1: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1835, 1, 96, 128);  mul_1835 = None
        add_tensor: "f32[64, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1853, 1, 96, 128);  mul_1853 = None
        add_tensor_1: "f32[64, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1871, 1, 96, 128);  mul_1871 = None
        add_tensor_2: "f32[64, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        le_scalar: "b8[64, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_self: "f32[64, 128, 56, 56]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_586);  le_scalar = full_default = getitem_586 = None
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[64, 128, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_1866);  cat_1 = unsqueeze_1866 = None
        mul_tensor: "f32[64, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 4.982461734693877e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 4.982461734693877e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_tensor_4: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_35);  squeeze_16 = primals_35 = None
        unsqueeze_default_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[64, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[64, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[64, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[64, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_tensor_4: "f32[64, 32, 56, 56]" = torch.ops.aten.slice.Tensor(mul_tensor_7, 1, 96, 128);  mul_tensor_7 = None
        add_tensor_3: "f32[64, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        return add_tensor_3


def _default_make_inputs():
    return [
    torch.randn([64, 256, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([64, 224, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([64, 192, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([64, 160, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
