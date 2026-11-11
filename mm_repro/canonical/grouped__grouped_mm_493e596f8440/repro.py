"""Curated MM repro. Pattern hash: 493e596f8440."""
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
    def forward(self, arg0_1: "bf16[1000, 32]", arg1_1: "bf16[1000, 2880]", arg2_1: "bf16[32, 2880, 5760]", arg3_1: "i32[32]", arg4_1: "bf16[32, 5760]"):
        # No stacktrace found for following nodes
        topk = torch.ops.aten.topk.default(arg0_1, 4);  arg0_1 = None
        getitem: "bf16[1000, 4]" = topk[0];  getitem = None
        getitem_1: "i64[1000, 4]" = topk[1];  topk = None
        view: "i64[4000]" = torch.ops.aten.view.default(getitem_1, [-1]);  getitem_1 = None
        sort = torch.ops.aten.sort.default(view);  view = None
        getitem_2: "i64[4000]" = sort[0]
        getitem_3: "i64[4000]" = sort[1];  sort = None
        ge: "b8[4000]" = torch.ops.aten.ge.Scalar(getitem_2, 32)
        unsqueeze: "b8[4000, 1]" = torch.ops.aten.unsqueeze.default(ge, -1);  ge = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div: "i64[4000]" = torch.ops.aten.div.Tensor_mode(getitem_3, 4, rounding_mode = 'floor');  getitem_3 = None
        index: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(arg1_1, [div]);  arg1_1 = div = None
        where: "bf16[4000, 2880]" = torch.ops.aten.where.self(unsqueeze, full, index);  unsqueeze = full = index = None
        _grouped_mm: "bf16[4000, 5760]" = torch.ops.aten._grouped_mm.default(where, arg2_1, arg3_1);  where = arg2_1 = arg3_1 = None
        clamp_max: "i64[4000]" = torch.ops.aten.clamp_max.default(getitem_2, 31);  getitem_2 = None
        index_1: "bf16[4000, 5760]" = torch.ops.aten.index.Tensor(arg4_1, [clamp_max]);  arg4_1 = clamp_max = None
        add: "bf16[4000, 5760]" = torch.ops.aten.add.Tensor(_grouped_mm, index_1);  _grouped_mm = index_1 = None
        slice_1: "bf16[4000, 2880]" = torch.ops.aten.slice.Tensor(add, 1, 1, 9223372036854775807, 2)
        convert_element_type: "f32[4000, 2880]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        clamp_min: "f32[4000, 2880]" = torch.ops.aten.clamp_min.default(convert_element_type, -7.0);  convert_element_type = None
        clamp_max_1: "f32[4000, 2880]" = torch.ops.aten.clamp_max.default(clamp_min, 7.0);  clamp_min = None
        convert_element_type_1: "bf16[4000, 2880]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.bfloat16);  clamp_max_1 = None
        add_1: "bf16[4000, 2880]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        slice_2: "bf16[4000, 2880]" = torch.ops.aten.slice.Tensor(add, 1, 0, 9223372036854775807, 2);  add = None
        convert_element_type_2: "f32[4000, 2880]" = torch.ops.prims.convert_element_type.default(slice_2, torch.float32);  slice_2 = None
        clamp_max_2: "f32[4000, 2880]" = torch.ops.aten.clamp_max.default(convert_element_type_2, 7.0);  convert_element_type_2 = None
        convert_element_type_3: "bf16[4000, 2880]" = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.bfloat16);  clamp_max_2 = None
        mul: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.702)
        sigmoid: "bf16[4000, 2880]" = torch.ops.aten.sigmoid.default(mul);  mul = None
        mul_1: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sigmoid);  convert_element_type_3 = sigmoid = None
        mul_2: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(add_1, mul_1);  add_1 = mul_1 = None
        return mul_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
