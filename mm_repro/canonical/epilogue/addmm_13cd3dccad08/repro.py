"""Curated MM repro. Pattern hash: 13cd3dccad08."""
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
    def forward(self, arg0_1: "bf16[2880, 32]", arg1_1: "bf16[32]", arg2_1: "bf16[1000, 2880]"):
        # No stacktrace found for following nodes
        addmm: "bf16[1000, 32]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, arg0_1);  arg1_1 = arg2_1 = arg0_1 = None
        topk = torch.ops.aten.topk.default(addmm, 4);  addmm = None
        getitem: "bf16[1000, 4]" = topk[0]
        getitem_1: "i64[1000, 4]" = topk[1];  topk = None
        view: "i64[4000]" = torch.ops.aten.view.default(getitem_1, [-1]);  getitem_1 = None
        sort = torch.ops.aten.sort.default(view);  view = None
        getitem_2: "i64[4000]" = sort[0]
        getitem_3: "i64[4000]" = sort[1];  sort = None
        ge: "b8[4000]" = torch.ops.aten.ge.Scalar(getitem_2, 32)
        convert_element_type: "f32[1000, 4]" = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        return (getitem_2, getitem_3, ge, convert_element_type)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
