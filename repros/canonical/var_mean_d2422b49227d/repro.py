"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train
Pattern hash: d2422b49227d
Shape hash: b53c5cc6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 1024], f32), T([49], i64, max=49), T([16, 512, 1024], f32), T([1024], f32), T([1024], f32), T([1024, 1024], f32), S([16, 512, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_143: "f32[8192, 1024]", inductor_seeds_default: "i64[49]", add_189: "f32[16, 512, 1024]", primals_391: "f32[1024]", primals_392: "f32[1024]", primals_393: "f32[1024, 1024]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_143, _shape_param_0);  addmm_143 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_tensor: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_189, mul_tensor_1);  add_189 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_391);  mul_tensor_2 = primals_391 = None
        add_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_392);  mul_tensor_3 = primals_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_393, [1, 0]);  primals_393 = None
        return (reshape_default_1, permute_default)


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
