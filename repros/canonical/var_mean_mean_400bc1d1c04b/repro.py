"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_train
Pattern hash: 400bc1d1c04b
Shape hash: 7ea204d9
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
    def forward(self, convolution_61: "f32[256, 960, 7, 7]", primals_308: "f32[960]", primals_309: "f32[960]", primals_310: "f32[1280, 960]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:211 in _forward_impl, code: x = self.features(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 960, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 960, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001);  getitem = None
        rsqrt_default: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_1);  convolution_61 = getitem_1 = None
        mul_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_308, -1);  primals_308 = None
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_309, -1);  primals_309 = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[256, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        add_tensor_2: "f32[256, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, 3)
        clamp_min_default: "f32[256, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_tensor_2, 0);  add_tensor_2 = None
        clamp_max_default: "f32[256, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_2: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor_1, clamp_max_default);  add_tensor_1 = clamp_max_default = None
        div_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_tensor_2, 6);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:213 in _forward_impl, code: x = self.avgpool(x)
        mean_dim: "f32[256, 960, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:214 in _forward_impl, code: x = torch.flatten(x, 1)
        reshape_default: "f32[256, 960]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:216 in _forward_impl, code: x = self.classifier(x)
        permute_default: "f32[960, 1280]" = torch.ops.aten.permute.default(primals_310, [1, 0]);  primals_310 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([256, 960, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([1280, 960], dtype=torch.float32, device='cuda'),
    [256, 960],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
