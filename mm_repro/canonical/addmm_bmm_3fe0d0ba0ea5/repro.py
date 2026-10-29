"""Curated MM repro. Pattern hash: 3fe0d0ba0ea5."""
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
    def forward(self, arg0_1: "bf16[8192, 4, 49, 32]", arg1_1: "bf16[32768, 56, 56]", arg2_1: "f32[128]", arg3_1: "f32[128, 128]", arg4_1: "i64[56]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        clone: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        constant_pad_nd: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_1);  view = _shape_param_1 = None
        bmm: "bf16[32768, 56, 32]" = torch.ops.aten.bmm.default(arg1_1, constant_pad_nd);  arg1_1 = constant_pad_nd = None
        slice_1: "bf16[32768, 49, 32]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -7);  bmm = None
        view_1: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(slice_1, _shape_param_2);  slice_1 = _shape_param_2 = None
        permute: "bf16[8192, 49, 4, 32]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone_1: "bf16[8192, 49, 4, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        convert_element_type: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        convert_element_type_1: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        view_3: "bf16[401408, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_3);  clone_1 = _shape_param_3 = None
        addmm: "bf16[401408, 128]" = torch.ops.aten.addmm.default(convert_element_type, view_3, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        view_6: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(addmm, _shape_param_4);  addmm = _shape_param_4 = None
        permute_2: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 2, 4, 5]);  view_6 = None
        clone_2: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_7: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_2, _shape_param_5);  clone_2 = _shape_param_5 = None
        index: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_7, [None, arg4_1]);  view_7 = None
        index_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index, [None, None, arg4_1]);  index = arg4_1 = None
        inductor_seeds: "i64[46]" = torch.ops.prims.inductor_seeds.default(46, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_6, inductor_lookup_seed, 'rand');  _shape_param_6 = inductor_lookup_seed = None
        lt: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random, 0.9956521736457944);  inductor_random = None
        convert_element_type_2: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16);  lt = None
        div: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_2, 0.9956521736457944);  convert_element_type_2 = None
        mul: "bf16[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_1, div);  index_1 = div = None
        return (view_3, mul)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
