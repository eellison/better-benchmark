"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: b699ca824f8e
Shape hash: 369f714a
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
_shapes_config = "(T([128, 2304], f32), T([128, 2304, 7, 7], f32, stride=(112896, 1, 16128, 2304)), S([128, 2304, 1, 1]), S([128, 2304, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 2304]", convolution_80: "f32[128, 2304, 7, 7]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 2304, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_dim: "f32[128, 2304, 1]" = torch.ops.aten.squeeze.dim(reshape_default, 3);  reshape_default = None
        squeeze_dim_1: "f32[128, 2304]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default: "f32[294912]" = torch.ops.aten.full.default([294912], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[294912]" = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, [128, 2304], [2304, 1], 0);  full_default = squeeze_dim_1 = None
        as_strided_default: "f32[128, 2304, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [128, 2304, 1, 1], [2304, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        div_scalar: "f32[128, 2304, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:569 in forward_features, code: x = self.final_act(x)
        neg_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.neg.default(convolution_80)
        exp_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(div_scalar, mul_tensor);  div_scalar = None
        sub_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_80, sub_tensor);  convolution_80 = sub_tensor = None
        add_tensor_1: "f32[128, 2304, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list: "f32[2304]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 2, 3]);  mul_tensor_3 = None
        return sum_dim_int_list



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
