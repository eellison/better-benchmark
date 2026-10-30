"""Curated MM repro. Pattern hash: e02ffaac377b."""
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
    def forward(self, arg0_1: "f32[288]", arg1_1: "f32[144, 288]", arg2_1: "bf16[131072, 144]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[288]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[144, 288]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[131072, 288]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        view: "bf16[512, 256, 288]" = torch.ops.aten.view.default(addmm, _shape_param_0);  _shape_param_0 = None
        convert_element_type_2: "f32[512, 256, 288]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        neg: "f32[512, 256, 288]" = torch.ops.aten.neg.default(convert_element_type_2)
        exp: "f32[512, 256, 288]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[512, 256, 288]" = torch.ops.aten.div.Tensor(convert_element_type_2, add);  convert_element_type_2 = add = None
        convert_element_type_3: "bf16[512, 256, 288]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view_1: "bf16[131072, 288]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_1);  convert_element_type_3 = _shape_param_1 = None
        return (addmm, view_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
