"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-8-9-linux.aws.h100_graph74
Pattern hash: 46b40033661d
Shape hash: 105c9900
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 3840], f64), T([127, 80], f64), T([127, 80], f64), S([1, 64, 64, 3840]), S([1, 4096, 3, 16, -1]), S([3, 16, 4096, 80]), S([16, 64, 64, 80]), S([64, 1024, 80]), S([64, 80, 64]), S([64, 1024, 80]), S([64, 80, 64]), S([1, 16, 4096, -1]), S([1, 16, 4096, -1]), S([1, 16, 4096, 80]), S([16, 4096, 80]), S([1, 16, 80, 4096]), S([16, 80, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_124: "f64[4096, 3840]", arg442_1: "f64[127, 80]", arg443_1: "f64[127, 80]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        view_default: "f64[1, 64, 64, 3840]" = torch.ops.aten.view.default(addmm_124, _shape_param_0);  addmm_124 = _shape_param_0 = None
        view_default_1: "f64[1, 4096, 3, 16, 80]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f64[3, 1, 16, 4096, 80]" = torch.ops.aten.permute.default(view_default_1, [2, 0, 3, 1, 4]);  view_default_1 = None
        view_default_2: "f64[3, 16, 4096, 80]" = torch.ops.aten.view.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        unbind_int = torch.ops.aten.unbind.int(view_default_2);  view_default_2 = None
        getitem: "f64[16, 4096, 80]" = unbind_int[0]
        getitem_1: "f64[16, 4096, 80]" = unbind_int[1]
        getitem_2: "f64[16, 4096, 80]" = unbind_int[2];  unbind_int = None
        iota_default: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[64, 1]" = torch.ops.aten.unsqueeze.default(iota_default, 1);  iota_default = None
        mul_tensor: "f32[64, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default, 1.0);  unsqueeze_default = None
        iota_default_1: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[1, 64]" = torch.ops.aten.unsqueeze.default(iota_default_1, 0);  iota_default_1 = None
        mul_tensor_1: "f32[1, 64]" = torch.ops.aten.mul.Tensor(unsqueeze_default_1, 1.0);  unsqueeze_default_1 = None
        sub_tensor: "f32[64, 64]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        add_tensor: "f32[64, 64]" = torch.ops.aten.add.Tensor(sub_tensor, 63.0);  sub_tensor = None
        convert_element_type_default: "i64[64, 64]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.int64);  add_tensor = None
        index_tensor: "f64[64, 64, 80]" = torch.ops.aten.index.Tensor(arg442_1, [convert_element_type_default]);  arg442_1 = convert_element_type_default = None
        iota_default_2: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_2: "i64[64, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, 1);  iota_default_2 = None
        mul_tensor_2: "f32[64, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_2, 1.0);  unsqueeze_default_2 = None
        iota_default_3: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_3: "i64[1, 64]" = torch.ops.aten.unsqueeze.default(iota_default_3, 0);  iota_default_3 = None
        mul_tensor_3: "f32[1, 64]" = torch.ops.aten.mul.Tensor(unsqueeze_default_3, 1.0);  unsqueeze_default_3 = None
        sub_tensor_1: "f32[64, 64]" = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        add_tensor_1: "f32[64, 64]" = torch.ops.aten.add.Tensor(sub_tensor_1, 63.0);  sub_tensor_1 = None
        convert_element_type_default_1: "i64[64, 64]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.int64);  add_tensor_1 = None
        index_tensor_1: "f64[64, 64, 80]" = torch.ops.aten.index.Tensor(arg443_1, [convert_element_type_default_1]);  arg443_1 = convert_element_type_default_1 = None
        view_default_3: "f64[16, 64, 64, 80]" = torch.ops.aten.view.default(getitem, _shape_param_3);  _shape_param_3 = None
        unsqueeze_default_4: "f64[16, 64, 64, 80, 1]" = torch.ops.aten.unsqueeze.default(view_default_3, 4)
        permute_default_1: "f64[16, 64, 64, 1, 80]" = torch.ops.aten.permute.default(unsqueeze_default_4, [0, 1, 2, 4, 3]);  unsqueeze_default_4 = None
        unsqueeze_default_5: "f64[64, 64, 80, 1]" = torch.ops.aten.unsqueeze.default(index_tensor, 3);  index_tensor = None
        unsqueeze_default_6: "f64[64, 64, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 4);  unsqueeze_default_5 = None
        permute_default_2: "f64[1, 64, 1, 64, 80]" = torch.ops.aten.permute.default(unsqueeze_default_6, [3, 0, 4, 1, 2]);  unsqueeze_default_6 = None
        permute_default_3: "f64[64, 16, 64, 80, 1]" = torch.ops.aten.permute.default(permute_default_1, [1, 0, 2, 4, 3]);  permute_default_1 = None
        clone_default: "f64[64, 16, 64, 80, 1]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_4: "f64[64, 1024, 80]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        permute_default_4: "f64[64, 80, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_2, [1, 4, 3, 0, 2]);  permute_default_2 = None
        view_default_5: "f64[64, 80, 64]" = torch.ops.aten.view.default(permute_default_4, _shape_param_5);  permute_default_4 = _shape_param_5 = None
        unsqueeze_default_7: "f64[16, 64, 64, 80, 1]" = torch.ops.aten.unsqueeze.default(view_default_3, 4);  view_default_3 = None
        permute_default_5: "f64[16, 64, 64, 1, 80]" = torch.ops.aten.permute.default(unsqueeze_default_7, [0, 1, 2, 4, 3]);  unsqueeze_default_7 = None
        unsqueeze_default_8: "f64[64, 64, 80, 1]" = torch.ops.aten.unsqueeze.default(index_tensor_1, 3);  index_tensor_1 = None
        unsqueeze_default_9: "f64[64, 64, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 4);  unsqueeze_default_8 = None
        permute_default_6: "f64[1, 1, 64, 64, 80]" = torch.ops.aten.permute.default(unsqueeze_default_9, [3, 4, 0, 1, 2]);  unsqueeze_default_9 = None
        permute_default_7: "f64[64, 16, 64, 80, 1]" = torch.ops.aten.permute.default(permute_default_5, [2, 0, 1, 4, 3]);  permute_default_5 = None
        clone_default_1: "f64[64, 16, 64, 80, 1]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        view_default_6: "f64[64, 1024, 80]" = torch.ops.aten.view.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_8: "f64[64, 80, 64, 1, 1]" = torch.ops.aten.permute.default(permute_default_6, [2, 4, 3, 0, 1]);  permute_default_6 = None
        view_default_7: "f64[64, 80, 64]" = torch.ops.aten.view.default(permute_default_8, _shape_param_7);  permute_default_8 = _shape_param_7 = None
        view_default_8: "f64[1, 16, 4096, 80]" = torch.ops.aten.view.default(getitem, _shape_param_8);  getitem = _shape_param_8 = None
        view_default_9: "f64[1, 16, 4096, 80]" = torch.ops.aten.view.default(getitem_1, _shape_param_9);  getitem_1 = _shape_param_9 = None
        mul_scalar: "f64[1, 16, 4096, 80]" = torch.ops.aten.mul.Scalar(view_default_8, 0.334370152488211);  view_default_8 = None
        permute_default_9: "f64[1, 16, 80, 4096]" = torch.ops.aten.permute.default(view_default_9, [0, 1, 3, 2]);  view_default_9 = None
        mul_scalar_1: "f64[1, 16, 80, 4096]" = torch.ops.aten.mul.Scalar(permute_default_9, 0.334370152488211);  permute_default_9 = None
        expand_default: "f64[1, 16, 4096, 80]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_10);  mul_scalar = _shape_param_10 = None
        view_default_10: "f64[16, 4096, 80]" = torch.ops.aten.view.default(expand_default, _shape_param_11);  expand_default = _shape_param_11 = None
        expand_default_1: "f64[1, 16, 80, 4096]" = torch.ops.aten.expand.default(mul_scalar_1, _shape_param_12);  mul_scalar_1 = _shape_param_12 = None
        view_default_11: "f64[16, 80, 4096]" = torch.ops.aten.view.default(expand_default_1, _shape_param_13);  expand_default_1 = _shape_param_13 = None
        return (view_default_4, view_default_5, view_default_6, view_default_7, view_default_10, view_default_11, getitem_2)


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
