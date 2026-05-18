"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: df860640df1d
Shape hash: 99b12f37
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_218: "f32[4, 80, 14, 14]", getitem_233: "f32[4, 80, 14, 14]", lt_2: "b8[4, 1, 1, 1]", convolution_34: "f32[4, 80, 14, 14]", unsqueeze_534: "f32[1, 80, 1, 1]", squeeze_61: "f32[80]", primals_154: "f32[80]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        add_tensor: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_218, getitem_233);  getitem_218 = getitem_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:41 in stochastic_depth, code: noise = noise.bernoulli_(survival_rate)
        convert_element_type_default: "f32[4, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_2, torch.float32);  lt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:43 in stochastic_depth, code: noise.div_(survival_rate)
        div_tensor: "f32[4, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.925);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/stochastic_depth.py:44 in stochastic_depth, code: return input * noise
        mul_tensor: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor, div_tensor);  add_tensor = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        sum_dim_int_list: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3])
        sub_tensor: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_534);  convolution_34 = unsqueeze_534 = None
        mul_tensor_1: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 2, 3]);  mul_tensor_1 = None
        mul_tensor_2: "f32[80]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0012755102040816326);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_2, 0);  mul_tensor_2 = None
        unsqueeze_default_1: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_3: "f32[80]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0012755102040816326);  sum_dim_int_list_1 = None
        mul_tensor_4: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_tensor_5: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        unsqueeze_default_3: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_4: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_6: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_154);  squeeze_61 = primals_154 = None
        unsqueeze_default_6: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_7: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_7: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_7);  mul_tensor = mul_tensor_7 = None
        sub_tensor_2: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_8: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_8


def _default_make_inputs():
    return [
    torch.randn(62720, dtype=torch.float32, device='cuda').as_strided([4, 80, 14, 14], [15680, 1, 1120, 80]),  # getitem_218
    torch.randn(62720, dtype=torch.float32, device='cuda').as_strided([4, 80, 14, 14], [15680, 1, 1120, 80]),  # getitem_233
    torch.randint(0, 2, [4, 1, 1, 1], dtype=torch.bool, device='cuda'),
    torch.randn(62720, dtype=torch.float32, device='cuda').as_strided([4, 80, 14, 14], [15680, 1, 1120, 80]),  # convolution_34
    torch.randn([1, 80, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
