"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: cfb663ddb1f4
Shape hash: 07f8d357
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 320, 8, 8], f32, stride=(20480, 1, 2560, 320)), T([320], f32), T([320], f32), T([128, 192, 8, 8], f32, stride=(12288, 1, 1536, 192)), T([192], f32), T([192], f32), T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)))"

class Repro(torch.nn.Module):
    def forward(self, convolution_71: "f32[128, 320, 8, 8]", primals_432: "f32[320]", primals_433: "f32[320]", convolution_75: "f32[128, 192, 8, 8]", primals_456: "f32[192]", primals_457: "f32[192]", cat_7: "f32[128, 768, 17, 17]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 320, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 320, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001);  getitem = None
        rsqrt_default: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_1);  convolution_71 = getitem_1 = None
        mul_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_432, -1);  primals_432 = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_433, -1);  primals_433 = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 320, 8, 8]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 192, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 192, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001);  getitem_2 = None
        rsqrt_default_1: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_3);  convolution_75 = getitem_3 = None
        mul_tensor_2: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_456, -1);  primals_456 = None
        unsqueeze_default_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_457, -1);  primals_457 = None
        unsqueeze_default_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 192, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:184 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False);  cat_7 = None
        getitem_4: "f32[128, 768, 8, 8]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_5: "i8[128, 768, 8, 8]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:190 in forward, code: return torch.cat(outputs, 1)
        cat_default: "f32[128, 1280, 8, 8]" = torch.ops.aten.cat.default([relu_default, relu_default_1, getitem_4], 1);  relu_default = relu_default_1 = getitem_4 = None
        return (cat_default, getitem_5)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
