"""Curated MM repro. Pattern hash: 2ef9cf25411a."""
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
    def forward(self, arg0_1: "bf16[192, 128, 64]", arg1_1: "bf16[192, 64, 128]", arg2_1: "bf16[32, 6]", arg3_1: "bf16[192, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 128]" = torch.ops.aten.view.default(iota, _shape_param_1);  iota = _shape_param_1 = None
        iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[128, 1]" = torch.ops.aten.view.default(iota_1, _shape_param_2);  iota_1 = _shape_param_2 = None
        add: "i64[128, 1]" = torch.ops.aten.add.Tensor(unsqueeze_1, 0);  unsqueeze_1 = None
        sub: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze, add);  unsqueeze = add = None
        gt: "b8[128, 128]" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul: "i64[128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
        add_1: "i64[128, 128]" = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        abs_1: "i64[128, 128]" = torch.ops.aten.abs.default(sub);  sub = None
        lt: "b8[128, 128]" = torch.ops.aten.lt.Scalar(abs_1, 8)
        convert_element_type_1: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
        log: "f32[128, 128]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_1: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
        convert_element_type_2: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        add_2: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None
        full: "i64[128, 128]" = torch.ops.aten.full.default(_shape_param_3, 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        minimum: "i64[128, 128]" = torch.ops.aten.minimum.default(add_2, full);  add_2 = full = None
        where: "i64[128, 128]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_3: "i64[128, 128]" = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
        embedding: "bf16[128, 128, 6]" = torch.ops.aten.embedding.default(arg2_1, add_3);  arg2_1 = add_3 = None
        permute: "bf16[6, 128, 128]" = torch.ops.aten.permute.default(embedding, [2, 0, 1]);  embedding = None
        unsqueeze_2: "bf16[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute, 0);  permute = None
        full_1: "bf16[32, 1, 128, 128]" = torch.ops.aten.full.default(_shape_param_4, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        add_4: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_2, full_1);  unsqueeze_2 = full_1 = None
        add_5: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view, add_4);  view = add_4 = None
        convert_element_type_3: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        amax: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_3, [3], True)
        sub_1: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_3, amax);  convert_element_type_3 = amax = None
        exp: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div_2: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_4: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        view_1: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_5);  convert_element_type_4 = _shape_param_5 = None
        bmm_1: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_1, arg3_1);  view_1 = arg3_1 = None
        view_2: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_6);  bmm_1 = _shape_param_6 = None
        permute_1: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_4: "bf16[4096, 384]" = torch.ops.aten.view.default(clone, _shape_param_7);  clone = _shape_param_7 = None
        return view_4

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
