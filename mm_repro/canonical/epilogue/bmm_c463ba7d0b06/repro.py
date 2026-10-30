"""Curated MM repro. Pattern hash: c463ba7d0b06."""
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
    def forward(self, arg0_1: "bf16[64, 1024, 64]", arg1_1: "bf16[64, 64, 1024]", arg2_1: "bf16[32, 8]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        bmm: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.view.default(iota, _shape_param_1);  iota = _shape_param_1 = None
        iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[1024, 1]" = torch.ops.aten.view.default(iota_1, _shape_param_2);  iota_1 = _shape_param_2 = None
        add: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze_1, 0);  unsqueeze_1 = None
        sub: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze, add);  unsqueeze = add = None
        full: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full);  sub = full = None
        neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum);  minimum = None
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type, 16);  convert_element_type = None
        log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.0794415416798357);  log = None
        mul: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 16);  div_1 = None
        convert_element_type_1: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
        add_1: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_1, 16);  convert_element_type_1 = None
        full_1: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_4, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_1, full_1);  add_1 = full_1 = None
        where: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, neg, minimum_1);  lt = neg = minimum_1 = None
        add_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where, 0);  where = None
        embedding: "bf16[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg2_1, add_2);  arg2_1 = add_2 = None
        permute: "bf16[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding, [2, 0, 1]);  embedding = None
        unsqueeze_2: "bf16[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute, 0);  permute = None
        iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[1024]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None
        unsqueeze_5: "i64[1, 1, 1, 1024]" = torch.ops.aten.view.default(add_3, _shape_param_5);  add_3 = _shape_param_5 = None
        iota_3: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[1024]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None
        unsqueeze_8: "i64[1, 1, 1024, 1]" = torch.ops.aten.view.default(add_4, _shape_param_6);  add_4 = _shape_param_6 = None
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_8);  unsqueeze_5 = unsqueeze_8 = None
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, _shape_param_7);  le = _shape_param_7 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_2, full_3);  expand = full_2 = full_3 = None
        add_5: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_2, where_1);  unsqueeze_2 = where_1 = None
        add_6: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view, add_5);  view = add_5 = None
        convert_element_type_2: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None
        return convert_element_type_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
