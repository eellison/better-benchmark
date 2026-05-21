"""
Standalone repro captured via capture_hook.
Label: torchbench_resnet152_train
Pattern hash: dfcf7a805f28
Shape hash: 20f1eeda
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
_shapes_config = "(T([32, 2048, 7, 7], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([2048], f32), T([32, 2048, 7, 7], f32), S([32, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_154: "f32[32, 2048, 7, 7]", primals_928: "f32[2048]", primals_929: "f32[2048]", primals_930: "f32[2048]", primals_931: "f32[2048]", relu_147: "f32[32, 2048, 7, 7]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_154, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 2048, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 2048, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_154, getitem_1);  convolution_154 = None
        mul_tensor: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_928, 0.9)
        add_tensor_1: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0006381620931717);  squeeze_dims_2 = None
        mul_tensor_4: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_929, 0.9)
        add_tensor_2: "f32[2048]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_930, -1);  primals_930 = None
        unsqueeze_default_1: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_931, -1);  primals_931 = None
        unsqueeze_default_3: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_tensor_4: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_3, relu_147);  add_tensor_3 = relu_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_default: "f32[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        mean_dim: "f32[32, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        reshape_default: "f32[32, 2048]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_scalar: "b8[32, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_default_4: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None

        # No stacktrace found for following nodes
        copy__default: "f32[2048]" = torch.ops.aten.copy_.default(primals_928, add_tensor_1);  primals_928 = add_tensor_1 = None
        copy__default_1: "f32[2048]" = torch.ops.aten.copy_.default(primals_929, add_tensor_2);  primals_929 = add_tensor_2 = None
        return (squeeze_dims_1, copy__default, unsqueeze_default_6, copy__default_1, le_scalar, reshape_default)



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
