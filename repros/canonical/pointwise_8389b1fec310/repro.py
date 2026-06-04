"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train_000
Pattern hash: 8389b1fec310
Shape hash: 04edfccc
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1024, 256, 13, 13], f32), S([1024, 9216]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_4: "f32[1024, 256, 13, 13]", _shape_param_0):
        # No stacktrace found for following nodes
        relu_default: "f32[1024, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], False)
        getitem: "f32[1024, 256, 6, 6]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[1024, 256, 6, 6]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        _adaptive_avg_pool2d_default: "f32[1024, 256, 6, 6]" = torch.ops.aten._adaptive_avg_pool2d.default(getitem, [6, 6]);  getitem = None
        view_default: "f32[1024, 9216]" = torch.ops.aten.view.default(_adaptive_avg_pool2d_default, _shape_param_0);  _adaptive_avg_pool2d_default = _shape_param_0 = None
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[1024, 9216]" = torch.ops.prims.inductor_random.default([1024, 9216], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[1024, 9216]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_tensor: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f32[1024, 9216]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        le_scalar: "b8[1024, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (getitem_1, mul_tensor_1, le_scalar)

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
