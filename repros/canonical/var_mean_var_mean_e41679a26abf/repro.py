"""
Standalone repro captured via capture_hook.
Label: torchbench_LearningToPaint_train
Pattern hash: e41679a26abf
Shape hash: cd3b9d56
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
_shapes_config = "(T([1024, 512, 4, 4], f32), T([512], f32), T([512], f32), T([1024, 512, 4, 4], f32), T([512], f32), T([512], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_17: "f32[1024, 512, 4, 4]", primals_108: "f32[512]", primals_109: "f32[512]", convolution_18: "f32[1024, 512, 4, 4]", primals_114: "f32[512]", primals_115: "f32[512]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_1);  convolution_17 = getitem_1 = None
        mul_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 512, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 512, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_3);  convolution_18 = getitem_3 = None
        mul_tensor_2: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
        unsqueeze_default_5: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_default_7: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        add_tensor_4: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(add_tensor_1, add_tensor_3);  add_tensor_1 = add_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_default: "f32[1024, 512, 4, 4]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        return relu_default



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
