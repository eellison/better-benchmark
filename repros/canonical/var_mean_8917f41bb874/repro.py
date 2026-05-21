"""
Standalone repro captured via capture_hook.
Label: torchbench_LearningToPaint_train
Pattern hash: 8917f41bb874
Shape hash: ebeac336
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
    def forward(self, convolution_20: "f32[1024, 512, 4, 4]", primals_126: "f32[512]", primals_127: "f32[512]", relu_14: "f32[1024, 512, 4, 4]", primals_128: "f32[65, 512]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_1);  convolution_20 = getitem_1 = None
        mul_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        add_tensor_2: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(add_tensor_1, relu_14);  add_tensor_1 = relu_14 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        relu_default: "f32[1024, 512, 4, 4]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:133 in forward, code: x = F.avg_pool2d(x, 4)
        avg_pool2d_default: "f32[1024, 512, 1, 1]" = torch.ops.aten.avg_pool2d.default(relu_default, [4, 4]);  relu_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:134 in forward, code: x = x.view(x.size(0), -1)
        reshape_default: "f32[1024, 512]" = torch.ops.aten.reshape.default(avg_pool2d_default, _shape_param_0);  avg_pool2d_default = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:135 in forward, code: x = self.fc(x)
        permute_default: "f32[512, 65]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024, 512, 4, 4], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512, 4, 4], dtype=torch.float32, device='cuda'),
    torch.randn([65, 512], dtype=torch.float32, device='cuda'),
    [1024, -1],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
