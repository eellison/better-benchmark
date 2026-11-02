"""Curated MM repro. Pattern hash: 745820755e0e."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[512]", arg1_1: "bf16[32768, 384]", arg2_1: "bf16[384, 512]", arg3_1: "i64[1, 512]", arg4_1: "bf16[512, 512]", arg5_1: "bf16[2, 512]", arg6_1: "bf16[512]", arg7_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        addmm: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg3_1, 1, 0, 128);  arg3_1 = None
        embedding: "bf16[1, 128, 512]" = torch.ops.aten.embedding.default(arg4_1, slice_1);  arg4_1 = slice_1 = None
        add: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view, embedding);  view = embedding = None
        full: "i64[256, 128]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        embedding_1: "bf16[256, 128, 512]" = torch.ops.aten.embedding.default(arg5_1, full);  arg5_1 = full = None
        add_1: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_1);  add = embedding_1 = None
        mul: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg6_1);  add_1 = arg6_1 = None
        add_2: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul, arg7_1);  mul = arg7_1 = None
        view_1: "bf16[32768, 512]" = torch.ops.aten.view.default(add_2, _shape_param_2);  _shape_param_2 = None
        view_2: "bf16[32768, 512]" = torch.ops.aten.view.default(add_2, _shape_param_3);  _shape_param_3 = None
        view_3: "bf16[32768, 512]" = torch.ops.aten.view.default(add_2, _shape_param_4);  _shape_param_4 = None
        return (add_2, view_1, view_2, view_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
