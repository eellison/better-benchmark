"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_train
Pattern hash: 13c233a8b548
Shape hash: 88deb8b3
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
    def forward(self, arg0_1: "f32[30522, 768]", arg1_1: "i64[32, 512]", arg2_1: "f32[512, 768]", arg3_1: "i64[1, 512]", arg4_1: "f32[1024, 768]", arg5_1: "f32[1024, 768]", arg6_1: "f32[1024, 768]", arg7_1: "f32[1024, 768]", arg8_1: "f32[2, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        full: "i64[32, 512]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        full_1: "i64[32, 512, 4]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        select: "i64[32, 512]" = torch.ops.aten.select.int(full_1, 2, 0)
        embedding_2: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select)
        select_1: "i64[32, 512]" = torch.ops.aten.select.int(full_1, 2, 1)
        embedding_3: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_1)
        select_2: "i64[32, 512]" = torch.ops.aten.select.int(full_1, 2, 2)
        embedding_4: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select_2);  arg4_1 = None
        select_3: "i64[32, 512]" = torch.ops.aten.select.int(full_1, 2, 3);  full_1 = None
        embedding_5: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_3);  arg5_1 = None
        sub: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_3, select_1)
        embedding_6: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg6_1, sub);  arg6_1 = None
        sub_1: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_2, select)
        embedding_7: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg7_1, sub_1);  arg7_1 = None
        embedding_8: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg8_1, full);  arg8_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_3);  add_1 = embedding_3 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_4);  add_2 = embedding_4 = None
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_3, embedding_5);  add_3 = embedding_5 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_4, embedding_6);  add_4 = embedding_6 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_5, embedding_7);  add_5 = embedding_7 = None
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_6, embedding_8);  add_6 = embedding_8 = None
        var_mean = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_8: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_1);  add_7 = getitem_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg9_1);  arg9_1 = None
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg10_1);  mul_1 = arg10_1 = None
        inductor_seeds: "i64[37]" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_9);  add_9 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16)
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_3);  convert_element_type = _shape_param_3 = None
        div: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (full, select, select_1, select_2, select_3, sub, sub_1, mul, inductor_seeds, gt, mul_3, view, div)



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
