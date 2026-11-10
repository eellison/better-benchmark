"""Curated MM repro. Pattern hash: 046b11fd0af8."""
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
    def forward(self, arg0_1: "bf16[192, 64, 512]", arg1_1: "bf16[192, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm, full);  bmm = full = None
        permute: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
        view: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        permute_1: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        view_1: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_1, _shape_param_1);  permute_1 = _shape_param_1 = None
        clone: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_1, memory_format = torch.contiguous_format);  view_1 = None
        view_2: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_2: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_1: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[1536]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_1: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (view_2, permute_2, convert_element_type_1)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
