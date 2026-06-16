"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: a955dfb90471
Shape hash: 4ea50a71
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
    def forward(self, arg0_1: "f32[20005, 768]", arg1_1: "i64[16, 128]", arg2_1: "f32[1, 512, 768]", arg3_1: "f32[3, 768]", arg4_1: "i64[16, 128]", arg5_1: "f32[768]", arg6_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding: "f32[16, 128, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        slice_1: "f32[1, 128, 768]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        add: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(embedding, slice_1);  embedding = slice_1 = None
        embedding_1: "f32[16, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, arg4_1, 0);  arg3_1 = arg4_1 = None
        add_1: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add, embedding_1);  add = embedding_1 = None
        inductor_seeds: "i64[61]" = torch.ops.prims.inductor_seeds.default(61, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[16, 128, 768]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[16, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_1: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        mean: "f32[16, 128, 1]" = torch.ops.aten.mean.dim(mul_1, [-1], True)
        var: "f32[16, 128, 1]" = torch.ops.aten.var.correction(mul_1, [-1], correction = 1.0, keepdim = True)
        sqrt: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None
        sub: "f32[16, 128, 768]" = torch.ops.aten.sub.Tensor(mul_1, mean);  mean = None
        mul_2: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg5_1, sub);  arg5_1 = None
        add_2: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06)
        div: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        add_3: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(div, arg6_1);  div = arg6_1 = None
        convert_element_type: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[2048, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        return (inductor_seeds, gt, mul_1, sqrt, sub, view)



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
