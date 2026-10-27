"""
Standalone repro captured via capture_hook.
Label: YituTechConvBert
Pattern hash: 659a141d567f
Shape hash: 2ec91973
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
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "f32[30522, 768]", arg2_1: "i64[32, 512]", arg3_1: "f32[512, 768]", arg4_1: "i64[1, 512]", arg5_1: "f32[2, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[384]", arg9_1: "f32[384, 768]", arg10_1: "f32[384]", arg11_1: "f32[384, 768]", arg12_1: "f32[384]", arg13_1: "f32[384, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg2_1, 0);  arg1_1 = arg2_1 = None
        embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        embedding_2: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, expand);  arg5_1 = expand = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  arg6_1 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        inductor_seeds: "i64[25]" = torch.ops.prims.inductor_seeds.default(25, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_3);  add_3 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type: "bf16[384]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        convert_element_type_1: "bf16[384, 768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_2: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16)
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        permute: "bf16[768, 384]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[16384, 384]" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm, _shape_param_3);  addmm = _shape_param_3 = None
        convert_element_type_3: "bf16[384]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        convert_element_type_4: "bf16[384, 768]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        permute_1: "bf16[768, 384]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm_1: "bf16[16384, 384]" = torch.ops.aten.addmm.default(convert_element_type_3, view, permute_1);  convert_element_type_3 = None
        view_2: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        permute_2: "f32[32, 768, 512]" = torch.ops.aten.permute.default(mul_3, [0, 2, 1])
        convert_element_type_5: "bf16[32, 768, 512]" = torch.ops.prims.convert_element_type.default(permute_2, torch.bfloat16);  permute_2 = None
        convert_element_type_6: "bf16[384]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        convert_element_type_7: "bf16[384, 768]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        permute_3: "bf16[768, 384]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_2: "bf16[16384, 384]" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_3);  convert_element_type_6 = None
        view_3: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm_2, _shape_param_5);  _shape_param_5 = None
        view_4: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_3, _shape_param_6);  _shape_param_6 = None
        view_5: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_1, _shape_param_7);  view_1 = _shape_param_7 = None
        view_6: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_2, _shape_param_8);  view_2 = _shape_param_8 = None
        permute_4: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_5: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        permute_6: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_7: "bf16[384, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_8: "bf16[384, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_9: "bf16[384, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        div: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (mul, inductor_seeds, gt, mul_3, view, convert_element_type_5, addmm_2, view_3, permute_4, permute_5, permute_6, permute_7, permute_8, permute_9, div)



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
