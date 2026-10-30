"""Curated MM repro. Pattern hash: fd90dcd8a7ba."""
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
    def forward(self, arg0_1: "f32[128]", arg1_1: "f32[512, 128]", arg2_1: "bf16[401408, 512]", arg3_1: "i64[46]", arg4_1: "f32[128, 3136, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[401408, 128]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        view: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 0);  arg3_1 = None
        inductor_random: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        lt: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random, 0.9956521736457944);  inductor_random = None
        convert_element_type_2: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16);  lt = None
        div: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_2, 0.9956521736457944);  convert_element_type_2 = None
        mul: "bf16[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view, div);  view = div = None
        add: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(arg4_1, mul);  arg4_1 = mul = None
        view_2: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        permute_1: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2, 5]);  view_2 = None
        clone: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "f32[128, 28, 28, 512]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        return view_3

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
