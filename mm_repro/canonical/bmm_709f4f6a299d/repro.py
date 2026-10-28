"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/T5ForConditionalGeneration/full_graph_000.py
Pattern hash: 709f4f6a299d
Shape hash: 0511b145
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 1024, 64]", arg1_1: "bf16[64, 64, 1024]", arg2_1: "f32[32, 8]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota, 1)
        add: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze, 0);  unsqueeze = None
        unsqueeze_1: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0)
        sub: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze_1, add);  unsqueeze_1 = add = None
        add_1: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_2: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_3: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        unsqueeze_4: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 3)
        unsqueeze_5: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_4);  unsqueeze_5 = unsqueeze_4 = None
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, _shape_param_0);  le = _shape_param_0 = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full, full_1);  expand = full = full_1 = None
        bmm: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        full_2: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full_2);  sub = full_2 = None
        neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum);  minimum = None
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type, 16);  convert_element_type = None
        log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.0794415416798357);  log = None
        mul: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 16);  div_1 = None
        convert_element_type_1: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
        add_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_1, 16);  convert_element_type_1 = None
        full_3: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_3, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_2, full_3);  add_2 = full_3 = None
        where_1: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, neg, minimum_1);  lt = neg = minimum_1 = None
        add_3: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_1, 0);  where_1 = None
        embedding: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg2_1, add_3);  arg2_1 = add_3 = None
        permute: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding, [2, 0, 1]);  embedding = None
        unsqueeze_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute, 0);  permute = None
        add_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_6, where);  unsqueeze_6 = where = None
        add_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view, add_4);  view = add_4 = None
        convert_element_type_2: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        convert_element_type_3: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        return convert_element_type_3



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
