"""Curated MM repro. Pattern hash: 572d071d157b."""
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
    def forward(self, arg0_1: "b8[16, 1, 128, 128]", arg1_1: "bf16[192, 128, 64]", arg2_1: "bf16[192, 64, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        eq: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(arg0_1, 0);  arg0_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        view: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        div: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view, 8.0);  view = None
        where: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq, full, div);  eq = full = div = None
        convert_element_type: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        return convert_element_type

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
