"""
Standalone repro captured via capture_hook.
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 512, 768], f16), T([2048, 768], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f16[4, 512, 768]", addmm_3: "f16[2048, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_default: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 512, 768]" = torch.ops.prims.inductor_random.default([4, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_1: "f16[4, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[4, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [4, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_tensor: "f16[4, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f16[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f16[4, 512, 768]" = torch.ops.aten.add.Tensor(arg2_1, mul_tensor_1);  arg2_1 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        reshape_default_1: "f16[2048, 768]" = torch.ops.aten.reshape.default(add_tensor, [-1, 768]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_default_2: "f32[2048, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [1], correction = 0, keepdim = True);  convert_element_type_default_2 = None
        getitem_2: "f32[2048, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[2048, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        return (getitem_2, getitem_3, getitem, getitem_1)


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
