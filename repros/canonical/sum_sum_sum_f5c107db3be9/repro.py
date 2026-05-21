"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: f5c107db3be9
Shape hash: f4823463
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
_shapes_config = "(T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([512, 512, 64], f32), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([8, 64, 512, 64]), S([8, 512, 4096]), S([4096, 4096]), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, view_30: "f32[4096, 4096]", view_60: "f32[4096, 4096]", view_90: "f32[4096, 4096]", view_120: "f32[4096, 4096]", view_150: "f32[4096, 4096]", view_180: "f32[4096, 4096]", view_210: "f32[4096, 4096]", view_240: "f32[4096, 4096]", view_270: "f32[4096, 4096]", view_300: "f32[4096, 4096]", view_330: "f32[4096, 4096]", bmm_44: "f32[512, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_30, [0], True);  view_30 = None
        view_default: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_60, [0], True);  view_60 = None
        view_default_1: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_90, [0], True);  view_90 = None
        view_default_2: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True);  view_120 = None
        view_default_3: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_3, _shape_param_3);  sum_dim_int_list_3 = _shape_param_3 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_3);  add_tensor_1 = view_default_3 = None
        sum_dim_int_list_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
        view_default_4: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_4);  sum_dim_int_list_4 = _shape_param_4 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, view_default_4);  add_tensor_2 = view_default_4 = None
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
        view_default_5: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_5, _shape_param_5);  sum_dim_int_list_5 = _shape_param_5 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, view_default_5);  add_tensor_3 = view_default_5 = None
        sum_dim_int_list_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
        view_default_6: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_6, _shape_param_6);  sum_dim_int_list_6 = _shape_param_6 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, view_default_6);  add_tensor_4 = view_default_6 = None
        sum_dim_int_list_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
        view_default_7: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, view_default_7);  add_tensor_5 = view_default_7 = None
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
        view_default_8: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_8, _shape_param_8);  sum_dim_int_list_8 = _shape_param_8 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, view_default_8);  add_tensor_6 = view_default_8 = None
        sum_dim_int_list_9: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
        view_default_9: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_9, _shape_param_9);  sum_dim_int_list_9 = _shape_param_9 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, view_default_9);  add_tensor_7 = view_default_9 = None
        sum_dim_int_list_10: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
        view_default_10: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_10, _shape_param_10);  sum_dim_int_list_10 = _shape_param_10 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, view_default_10);  add_tensor_8 = view_default_10 = None
        view_default_11: "f32[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_11);  bmm_44 = _shape_param_11 = None
        permute_default: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_default_11, [0, 2, 1, 3]);  view_default_11 = None
        clone_default: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_12: "f32[8, 512, 4096]" = torch.ops.aten.view.default(clone_default, _shape_param_12);  clone_default = _shape_param_12 = None
        view_default_13: "f32[4096, 4096]" = torch.ops.aten.view.default(view_default_12, _shape_param_13);  view_default_12 = _shape_param_13 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_default_13, [1, 0])
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_default_13, [0], True);  view_default_13 = None
        view_default_14: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_11, _shape_param_14);  sum_dim_int_list_11 = _shape_param_14 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, view_default_14);  add_tensor_9 = view_default_14 = None
        return (permute_default_1, add_tensor_10)



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
