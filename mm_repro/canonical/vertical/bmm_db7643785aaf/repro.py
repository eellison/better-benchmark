"""Curated MM repro. Pattern hash: db7643785aaf."""
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
    def forward(self, arg0_1: "f32[512, 128, 128]", arg1_1: "f32[512, 128, 128]", arg2_1: "f32[32, 16, 128, 128]", arg3_1: "f32[]", arg4_1: "bf16[512, 64, 128]", arg5_1: "bf16[512, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        sum_1: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [2], True)
        neg: "f32[512, 128, 128]" = torch.ops.aten.neg.default(arg1_1);  arg1_1 = None
        fma: "f32[512, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, arg0_1);  neg = sum_1 = arg0_1 = None
        view: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(fma, _shape_param_0);  fma = _shape_param_0 = None
        div: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Scalar(view, 2)
        eq: "b8[32, 16, 128, 128]" = torch.ops.aten.eq.Tensor(arg2_1, arg3_1)
        where: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(eq, div, view);  eq = div = view = None
        lt: "b8[32, 16, 128, 128]" = torch.ops.aten.lt.Tensor(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(lt, full, where);  lt = full = where = None
        convert_element_type: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        bmm: "bf16[512, 64, 128]" = torch.ops.aten.bmm.default(arg4_1, view_1);  arg4_1 = None
        bmm_1: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(view_1, arg5_1);  view_1 = arg5_1 = None
        permute: "bf16[512, 128, 64]" = torch.ops.aten.permute.default(bmm, [0, 2, 1]);  bmm = None
        view_2: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        view_3: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_3);  bmm_1 = _shape_param_3 = None
        permute_1: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        clone: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_4: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        permute_2: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_5: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(permute_2, _shape_param_5);  permute_2 = _shape_param_5 = None
        clone_1: "bf16[32, 128, 1024]" = torch.ops.aten.clone.default(view_5, memory_format = torch.contiguous_format);  view_5 = None
        view_6: "bf16[4096, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        permute_3: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_6, [1, 0])
        mul: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_4, 0.125);  view_4 = None
        view_7: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul, _shape_param_7);  mul = _shape_param_7 = None
        permute_4: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_7, [1, 0])
        return (view_6, permute_3, view_7, permute_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
