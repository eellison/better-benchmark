"""Curated MM repro. Pattern hash: f1fae524f4fd."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "bf16[32000, 1024]", arg2_1: "bf16[1024, 16, 64]", arg3_1: "bf16[16, 64]", arg4_1: "bf16[16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        permute: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding: "bf16[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone);  arg1_1 = clone = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.view.default(embedding, _shape_param_0);  embedding = _shape_param_0 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze = squeeze_1 = None
        view_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg3_1);  arg3_1 = None
        unsqueeze_5: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        permute_2: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_5, [1, 2, 0, 4, 3]);  unsqueeze_5 = None
        permute_3: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_2, [0, 1, 2, 4, 3]);  permute_2 = None
        view_4: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_5);  permute_3 = _shape_param_5 = None
        add_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg4_1);  view_3 = arg4_1 = None
        unsqueeze_6: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add_1, _shape_param_6);  add_1 = _shape_param_6 = None
        permute_4: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_6, [1, 2, 0, 4, 3]);  unsqueeze_6 = None
        permute_5: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_4, [0, 1, 2, 4, 3]);  permute_4 = None
        view_5: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_5, _shape_param_7);  permute_5 = _shape_param_7 = None
        return (view_4, view_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
