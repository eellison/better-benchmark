"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train
Pattern hash: 8187f7c4e01e
Shape hash: ad62e054
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
    def forward(self, arg0_1: "f32[29056, 1024]", arg1_1: "i64[16, 512]", arg2_1: "f32[2, 1024]", arg3_1: "f32[512, 1024]", arg4_1: "i64[1, 512]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full: "i64[16, 512]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        embedding: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        embedding_1: "f32[16, 512, 1024]" = torch.ops.aten.embedding.default(arg2_1, full);  arg2_1 = None
        add: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2: "f32[1, 512, 1024]" = torch.ops.aten.embedding.default(arg3_1, arg4_1);  arg3_1 = arg4_1 = None
        add_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        inductor_seeds: "i64[49]" = torch.ops.prims.inductor_seeds.default(49, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        gt: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        var_mean = torch.ops.aten.var_mean.correction(mul_1, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean[0]
        getitem_1: "f32[16, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1, getitem_1);  getitem_1 = None
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  arg5_1 = None
        add_3: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_3, arg6_1);  mul_3 = arg6_1 = None
        convert_element_type: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        div: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        return (full, inductor_seeds, gt, mul_1, mul_2, view, div)



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
