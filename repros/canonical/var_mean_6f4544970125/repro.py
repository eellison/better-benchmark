"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train
Pattern hash: 6f4544970125
Shape hash: 2777e7f5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 1024], f32), T([1024], f32), T([1024], f32), T([29056, 1024], f32), S([16, 512, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_144: "f32[8192, 1024]", primals_395: "f32[1024]", primals_396: "f32[1024]", primals_3: "f32[29056, 1024]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_144, _shape_param_0);  addmm_144 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[16, 512, 1024]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(mul_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_2, getitem_1);  mul_tensor_2 = getitem_1 = None
        mul_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, primals_395);  mul_tensor_3 = primals_395 = None
        add_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, primals_396);  mul_tensor_4 = primals_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[1024, 29056]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
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
