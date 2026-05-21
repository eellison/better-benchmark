"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_GPT2_large_train
Pattern hash: df4bf3b891c2
Shape hash: a84e9813
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 1280], f32), T([73], i64), T([4, 512, 1280], f32), T([1280], f32), T([1280], f32), S([4, 512, 1280]), S([-1, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_141: "f32[2048, 1280]", inductor_seeds_default: "i64[73]", add_283: "f32[4, 512, 1280]", primals_430: "f32[1280]", primals_431: "f32[1280]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(addmm_141, _shape_param_0);  addmm_141 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 71);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 512, 1280]" = torch.ops.prims.inductor_random.default([4, 512, 1280], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        add_tensor: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_tensor_1, add_283);  mul_tensor_1 = add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_430);  mul_tensor_2 = primals_430 = None
        add_tensor_2: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_431);  mul_tensor_3 = primals_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_1: "f32[2048, 1280]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        return reshape_default_1



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
