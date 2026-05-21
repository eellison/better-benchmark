"""
Standalone repro captured via capture_hook.
Label: torchbench_doctr_reco_predictor_infer_000
Pattern hash: 7727cefe098e
Shape hash: d8a1a12f
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
_shapes_config = "(T([512], f32), T([1, 512, 2, 32], f32), T([512], f32), T([512], f32), T([512], f32), S([-1, 512, 32]))"

class Repro(torch.nn.Module):
    def forward(self, arg75_1: "f32[512]", convolution_12: "f32[1, 512, 2, 32]", arg76_1: "f32[512]", arg77_1: "f32[512]", arg78_1: "f32[512]", _shape_param_0):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[1, 512, 2, 32]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_default_1);  convolution_12 = unsqueeze_default_1 = None
        add_tensor: "f32[512]" = torch.ops.aten.add.Tensor(arg76_1, 1e-05);  arg76_1 = None
        sqrt_default: "f32[512]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[1, 512, 2, 32]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_default_5: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[1, 512, 2, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_default_7: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[1, 512, 2, 32]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        relu_default: "f32[1, 512, 2, 32]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [2, 1], [2, 1], [0, 0], [1, 1], False);  relu_default = None
        getitem: "f32[1, 512, 1, 32]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[1, 512, 1, 32]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        view_default: "f32[1, 512, 32]" = torch.ops.aten.view.default(getitem, _shape_param_0);  getitem = _shape_param_0 = None
        permute_default: "f32[1, 32, 512]" = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
        return (getitem_1, permute_default)



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
