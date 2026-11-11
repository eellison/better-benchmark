"""Curated MM repro. Pattern hash: 1528b3b85610."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "bf16[32000, 1024]", arg2_1: "bf16[1024, 16, 64]", arg3_1: "bf16[16, 64]", arg4_1: "bf16[1024, 16, 64]", arg5_1: "bf16[16, 64]", arg6_1: "bf16[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        permute: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding: "bf16[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone);  arg1_1 = clone = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.view.default(embedding, _shape_param_0);  _shape_param_0 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze = squeeze_1 = None
        view_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg3_1);  arg3_1 = None
        squeeze_2: "bf16[8192, 1024]" = torch.ops.aten.view.default(embedding, _shape_param_4);  _shape_param_4 = None
        squeeze_3: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg4_1, _shape_param_5);  arg4_1 = _shape_param_5 = None
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_2, squeeze_3);  squeeze_2 = squeeze_3 = None
        add_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg5_1);  view_3 = arg5_1 = None
        squeeze_4: "bf16[8192, 1024]" = torch.ops.aten.view.default(embedding, _shape_param_6);  embedding = _shape_param_6 = None
        squeeze_5: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg6_1, _shape_param_7);  arg6_1 = _shape_param_7 = None
        mm_2: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_5);  squeeze_4 = squeeze_5 = None
        return (add, mm_1, add_1, mm_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
