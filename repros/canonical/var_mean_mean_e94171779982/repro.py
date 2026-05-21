"""
Standalone repro captured via capture_hook.
Label: torchbench_phlippe_resnet_train
Pattern hash: e94171779982
Shape hash: cf01f6e7
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
_shapes_config = "(T([128, 64, 8, 8], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([128, 64, 8, 8], f32), S([128, 64]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_20: "f32[128, 64, 8, 8]", primals_116: "f32[64]", primals_117: "f32[64]", primals_118: "f32[64]", primals_119: "f32[64]", relu_16: "f32[128, 64, 8, 8]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 64, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_1);  convolution_20 = None
        mul_tensor: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_116, 0.9)
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0001220852154804);  squeeze_dims_2 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1);  primals_119 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:54 in forward, code: out = z + x
        add_tensor_4: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(add_tensor_3, relu_16);  add_tensor_3 = relu_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        relu_default: "f32[128, 64, 8, 8]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:139 in forward, code: x = self.output_net(x)
        mean_dim: "f32[128, 64, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True)
        reshape_default: "f32[128, 64]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:55 in forward, code: out = self.act_fn(out)
        le_scalar: "b8[128, 64, 8, 8]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_resnet/__init__.py:51 in forward, code: z = self.net(x)
        unsqueeze_default_4: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None

        # No stacktrace found for following nodes
        copy__default: "f32[64]" = torch.ops.aten.copy_.default(primals_116, add_tensor_1);  primals_116 = add_tensor_1 = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(primals_117, add_tensor_2);  primals_117 = add_tensor_2 = None
        return (squeeze_dims_1, unsqueeze_default_6, copy__default, copy__default_1, le_scalar, reshape_default)



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
