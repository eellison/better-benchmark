"""Curated MM repro. Pattern hash: 94f85203ee57."""
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
    def forward(self, arg0_1: "bf16[30522, 128]", arg1_1: "i64[256, 128]", arg2_1: "bf16[384, 512]", arg3_1: "bf16[512]", arg4_1: "i64[1, 512]", arg5_1: "bf16[512, 512]", arg6_1: "bf16[2, 512]", arg7_1: "bf16[512]", arg8_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        embedding: "bf16[256, 128, 128]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        slice_1: "bf16[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 1, 9223372036854775807)
        constant_pad_nd: "bf16[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_1, [0, 0, 0, 1, 0, 0], 0.0);  slice_1 = None
        slice_2: "bf16[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 0, -1)
        constant_pad_nd_1: "bf16[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_2, [0, 0, 1, 0, 0, 0], 0.0);  slice_2 = None
        cat: "bf16[256, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None
        view: "bf16[32768, 384]" = torch.ops.aten.view.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        addmm: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg3_1, view, arg2_1);  arg3_1 = view = arg2_1 = None
        view_1: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        slice_3: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg4_1, 1, 0, 128);  arg4_1 = None
        embedding_1: "bf16[1, 128, 512]" = torch.ops.aten.embedding.default(arg5_1, slice_3);  arg5_1 = slice_3 = None
        add: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None
        full: "i64[256, 128]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        embedding_2: "bf16[256, 128, 512]" = torch.ops.aten.embedding.default(arg6_1, full);  arg6_1 = full = None
        add_1: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        mul: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg7_1);  add_1 = arg7_1 = None
        add_2: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul, arg8_1);  mul = arg8_1 = None
        view_2: "bf16[32768, 512]" = torch.ops.aten.view.default(add_2, _shape_param_3);  _shape_param_3 = None
        view_3: "bf16[32768, 512]" = torch.ops.aten.view.default(add_2, _shape_param_4);  _shape_param_4 = None
        view_4: "bf16[32768, 512]" = torch.ops.aten.view.default(add_2, _shape_param_5);  _shape_param_5 = None
        return (add_2, view_2, view_3, view_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
