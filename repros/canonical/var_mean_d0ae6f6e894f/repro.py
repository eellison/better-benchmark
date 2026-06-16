"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_infer
Pattern hash: d0ae6f6e894f
Shape hash: 771dc86a
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
    def forward(self, arg0_1: "bf16[30522, 768]", arg1_1: "i64[32, 512]", arg2_1: "bf16[512, 768]", arg3_1: "i64[1, 512]", arg4_1: "bf16[1024, 768]", arg5_1: "bf16[1024, 768]", arg6_1: "bf16[1024, 768]", arg7_1: "bf16[1024, 768]", arg8_1: "bf16[2, 768]", arg9_1: "bf16[768]", arg10_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        embedding: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        embedding_1: "bf16[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        add: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        full: "i64[32, 512, 4]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        select: "i64[32, 512]" = torch.ops.aten.select.int(full, 2, 0)
        embedding_2: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select);  select = None
        add_1: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        select_1: "i64[32, 512]" = torch.ops.aten.select.int(full, 2, 1)
        embedding_3: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_1);  select_1 = None
        add_2: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_3);  add_1 = embedding_3 = None
        select_2: "i64[32, 512]" = torch.ops.aten.select.int(full, 2, 2)
        embedding_4: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select_2);  arg4_1 = select_2 = None
        add_3: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_4);  add_2 = embedding_4 = None
        select_3: "i64[32, 512]" = torch.ops.aten.select.int(full, 2, 3)
        embedding_5: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_3);  arg5_1 = select_3 = None
        add_4: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add_3, embedding_5);  add_3 = embedding_5 = None
        select_4: "i64[32, 512]" = torch.ops.aten.select.int(full, 2, 3)
        select_5: "i64[32, 512]" = torch.ops.aten.select.int(full, 2, 1)
        sub: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_4, select_5);  select_4 = select_5 = None
        embedding_6: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg6_1, sub);  arg6_1 = sub = None
        add_5: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add_4, embedding_6);  add_4 = embedding_6 = None
        select_6: "i64[32, 512]" = torch.ops.aten.select.int(full, 2, 2)
        select_7: "i64[32, 512]" = torch.ops.aten.select.int(full, 2, 0);  full = None
        sub_1: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_6, select_7);  select_6 = select_7 = None
        embedding_7: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg7_1, sub_1);  arg7_1 = sub_1 = None
        add_6: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add_5, embedding_7);  add_5 = embedding_7 = None
        full_1: "i64[32, 512]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        embedding_8: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg8_1, full_1);  arg8_1 = full_1 = None
        add_7: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add_6, embedding_8);  add_6 = embedding_8 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_8: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg9_1);  mul = arg9_1 = None
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg10_1);  mul_1 = arg10_1 = None
        convert_element_type_1: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  _shape_param_2 = None
        view_1: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  _shape_param_3 = None
        view_2: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_4);  _shape_param_4 = None
        return (convert_element_type_1, view, view_1, view_2)



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
