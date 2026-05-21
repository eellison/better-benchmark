"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train
Pattern hash: 7b9458109f5a
Shape hash: 59a882c4
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
_shapes_config = "(T([64, 1024, 7, 7], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), S([64, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, cat_57: "f32[64, 1024, 7, 7]", primals_723: "f32[1024]", primals_724: "f32[1024]", primals_725: "f32[1024]", primals_726: "f32[1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_57, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 1024, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1024, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[64, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(cat_57, getitem_1);  cat_57 = None
        mul_tensor: "f32[64, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_723, 0.9)
        add_tensor_1: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0003189792663476);  squeeze_dims_1 = None
        mul_tensor_4: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_724, 0.9)
        add_tensor_2: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_725, -1);  primals_725 = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[64, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_726, -1);  primals_726 = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[64, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:214 in forward, code: out = F.relu(features, inplace=True)
        relu_default: "f32[64, 1024, 7, 7]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:215 in forward, code: out = F.adaptive_avg_pool2d(out, (1, 1))
        mean_dim: "f32[64, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:216 in forward, code: out = torch.flatten(out, 1)
        reshape_default: "f32[64, 1024]" = torch.ops.aten.reshape.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None

        # No stacktrace found for following nodes
        copy__default: "f32[1024]" = torch.ops.aten.copy_.default(primals_723, add_tensor_1);  primals_723 = add_tensor_1 = None
        copy__default_1: "f32[1024]" = torch.ops.aten.copy_.default(primals_724, add_tensor_2);  primals_724 = add_tensor_2 = None
        return (copy__default, copy__default_1, reshape_default)



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
