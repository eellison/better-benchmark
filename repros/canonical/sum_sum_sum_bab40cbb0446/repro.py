"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: bab40cbb0446
Shape hash: 87255a55
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
_shapes_config = "(T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([512, 64, 512], f32), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([8, 64, 64, 512]), S([8, 512, 4096]), S([4096, 4096]), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, view_34: "f32[4096, 4096]", view_64: "f32[4096, 4096]", view_94: "f32[4096, 4096]", view_124: "f32[4096, 4096]", view_154: "f32[4096, 4096]", view_184: "f32[4096, 4096]", view_214: "f32[4096, 4096]", view_244: "f32[4096, 4096]", view_274: "f32[4096, 4096]", view_304: "f32[4096, 4096]", view_334: "f32[4096, 4096]", bmm_46: "f32[512, 64, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True);  view_34 = None
        view_default: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_64, [0], True);  view_64 = None
        view_default_1: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_94, [0], True);  view_94 = None
        view_default_2: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True);  view_124 = None
        view_default_3: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_3, _shape_param_3);  sum_dim_int_list_3 = _shape_param_3 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_3);  add_tensor_1 = view_default_3 = None
        sum_dim_int_list_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
        view_default_4: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_4);  sum_dim_int_list_4 = _shape_param_4 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, view_default_4);  add_tensor_2 = view_default_4 = None
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_184, [0], True);  view_184 = None
        view_default_5: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_5, _shape_param_5);  sum_dim_int_list_5 = _shape_param_5 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, view_default_5);  add_tensor_3 = view_default_5 = None
        sum_dim_int_list_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True);  view_214 = None
        view_default_6: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_6, _shape_param_6);  sum_dim_int_list_6 = _shape_param_6 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, view_default_6);  add_tensor_4 = view_default_6 = None
        sum_dim_int_list_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True);  view_244 = None
        view_default_7: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, view_default_7);  add_tensor_5 = view_default_7 = None
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
        view_default_8: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_8, _shape_param_8);  sum_dim_int_list_8 = _shape_param_8 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, view_default_8);  add_tensor_6 = view_default_8 = None
        sum_dim_int_list_9: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        view_default_9: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_9, _shape_param_9);  sum_dim_int_list_9 = _shape_param_9 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, view_default_9);  add_tensor_7 = view_default_9 = None
        sum_dim_int_list_10: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
        view_default_10: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_10, _shape_param_10);  sum_dim_int_list_10 = _shape_param_10 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, view_default_10);  add_tensor_8 = view_default_10 = None
        view_default_11: "f32[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_46, _shape_param_11);  bmm_46 = _shape_param_11 = None
        mul_scalar: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_11, 0.3535533905932738);  view_default_11 = None
        permute_default: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        permute_default_1: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_12: "f32[8, 512, 4096]" = torch.ops.aten.view.default(permute_default_1, _shape_param_12);  permute_default_1 = _shape_param_12 = None
        clone_default: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(view_default_12, memory_format = torch.contiguous_format);  view_default_12 = None
        view_default_13: "f32[4096, 4096]" = torch.ops.aten.view.default(clone_default, _shape_param_13);  clone_default = _shape_param_13 = None
        permute_default_2: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_default_13, [1, 0])
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_default_13, [0], True);  view_default_13 = None
        view_default_14: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_11, _shape_param_14);  sum_dim_int_list_11 = _shape_param_14 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, view_default_14);  add_tensor_9 = view_default_14 = None
        return (permute_default_2, add_tensor_10)



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
