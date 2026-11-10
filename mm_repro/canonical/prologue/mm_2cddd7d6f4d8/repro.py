"""Curated MM repro. Pattern hash: 2cddd7d6f4d8."""
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
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "bf16[32000, 1024]", arg2_1: "bf16[1024, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        permute: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding: "bf16[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone);  arg1_1 = clone = None
        unsqueeze: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding, 3);  embedding = None
        unsqueeze_1: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 4);  unsqueeze = None
        view: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, arg2_1);  squeeze = arg2_1 = None
        return mm

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
