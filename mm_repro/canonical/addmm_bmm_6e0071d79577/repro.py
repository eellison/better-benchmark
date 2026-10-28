"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/infer/swin_base_patch4_window7_224/full_graph_000.py
Pattern hash: 6e0071d79577
Shape hash: b63cf89e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32768, 56, 56]", arg1_1: "bf16[32768, 56, 32]", arg2_1: "bf16[128, 128]", arg3_1: "bf16[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        bmm: "bf16[32768, 56, 32]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "bf16[32768, 49, 32]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -7);  bmm = None
        view: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        permute: "bf16[8192, 49, 4, 32]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        clone: "bf16[8192, 49, 4, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[401408, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_1: "bf16[128, 128]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm: "bf16[401408, 128]" = torch.ops.aten.addmm.default(arg3_1, view_2, permute_1);  arg3_1 = view_2 = permute_1 = None
        view_3: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(addmm, _shape_param_3);  addmm = _shape_param_3 = None
        view_4: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        view_5: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_2: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_5, [0, 1, 3, 2, 4, 5]);  view_5 = None
        clone_1: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_6: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        iota: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[56]" = torch.ops.aten.add.Tensor(iota, 53);  iota = None
        fmod: "i64[56]" = torch.ops.aten.fmod.Scalar(add, 56);  add = None
        index: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_6, [None, fmod]);  view_6 = fmod = None
        iota_1: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[56]" = torch.ops.aten.add.Tensor(iota_1, 53);  iota_1 = None
        fmod_1: "i64[56]" = torch.ops.aten.fmod.Scalar(add_1, 56);  add_1 = None
        index_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index, [None, None, fmod_1]);  index = fmod_1 = None
        return index_1



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
