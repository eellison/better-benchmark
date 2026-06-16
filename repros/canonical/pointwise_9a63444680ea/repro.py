"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 9a63444680ea
Shape hash: edd7640e
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
    def forward(self, arg0_1: "bf16[16384, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[32, 512, 384]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[32, 384, 512]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        clone: "bf16[32, 384, 512]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        unsqueeze: "bf16[32, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone, -1);  clone = None
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        add: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        full: "i64[1, 1]" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        constant_pad_nd: "bf16[32, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze, _shape_param_1, 0.0);  unsqueeze = _shape_param_1 = None
        unsqueeze_3: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add, -1);  add = None
        unsqueeze_4: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, -1);  unsqueeze_3 = None
        index: "bf16[32, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd, [None, None, unsqueeze_4, full]);  constant_pad_nd = None
        permute_1: "bf16[32, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
        view_1: "bf16[32, 3456, 512]" = torch.ops.aten.view.default(permute_1, _shape_param_2);  permute_1 = _shape_param_2 = None
        permute_2: "bf16[32, 512, 3456]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        view_2: "bf16[32, 512, 384, 9]" = torch.ops.aten.view.default(permute_2, _shape_param_3);  permute_2 = _shape_param_3 = None
        clone_1: "bf16[32, 512, 384, 9]" = torch.ops.aten.clone.default(view_2, memory_format = torch.contiguous_format);  view_2 = None
        view_3: "bf16[98304, 64, 9]" = torch.ops.aten.view.default(clone_1, _shape_param_4);  clone_1 = _shape_param_4 = None
        expand: "bf16[98304, 64, 9]" = torch.ops.aten.expand.default(view_3, _shape_param_5);  view_3 = _shape_param_5 = None
        permute_3: "bf16[98304, 9, 64]" = torch.ops.aten.permute.default(expand, [0, 2, 1])
        return (full, unsqueeze_4, expand, permute_3)



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
