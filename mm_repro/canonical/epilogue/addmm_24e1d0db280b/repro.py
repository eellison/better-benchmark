"""Curated MM repro. Pattern hash: 24e1d0db280b."""
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
    def forward(self, arg0_1: "f32[128]", arg1_1: "f32[128, 128]", arg2_1: "bf16[401408, 128]", arg3_1: "i64[56]", arg4_1: "f32[128, 56, 56, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        addmm: "bf16[401408, 128]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, convert_element_type_1);  convert_element_type = arg2_1 = convert_element_type_1 = None
        view_2: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        add: "i64[56]" = torch.ops.aten.add.Tensor(arg3_1, 53);  arg3_1 = None
        fmod: "i64[56]" = torch.ops.aten.fmod.Scalar(add, 56);  add = None
        index: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_3, [None, fmod]);  view_3 = None
        index_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index, [None, None, fmod]);  index = fmod = None
        inductor_seeds: "i64[46]" = torch.ops.prims.inductor_seeds.default(46, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        lt: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random, 0.9956521736457944);  inductor_random = None
        convert_element_type_2: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16);  lt = None
        div: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_2, 0.9956521736457944);  convert_element_type_2 = None
        mul: "bf16[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_1, div);  index_1 = div = None
        add_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(arg4_1, mul);  arg4_1 = mul = None
        view_4: "f32[128, 3136, 128]" = torch.ops.aten.view.default(add_1, _shape_param_3);  add_1 = _shape_param_3 = None
        return view_4

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
