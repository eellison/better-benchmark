"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_training
Pattern hash: e37108342905
Shape hash: 68c6dbc3
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_3: "f32[8192, 768]", _shape_param_0, primals_3: "f32[4, 2048, 768]", _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_0);  addmm_3 = _shape_param_0 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 2048, 768]" = torch.ops.prims.inductor_random.default([4, 2048, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 2048, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(primals_3, mul_tensor_1);  primals_3 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_1, [1], correction = 0, keepdim = True);  reshape_default_1 = None
        getitem: "f32[8192, 1]" = var_mean_correction[0]
        getitem_1: "f32[8192, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [4, 2048, 768],  # _shape_param_0
    torch.randn([4, 2048, 768], dtype=torch.float32, device='cuda'),
    [-1, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
