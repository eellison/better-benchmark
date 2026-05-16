"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, inductor_seeds_default: "i64[2]", addmm_5: "f16[2048, 768]", view_12: "f16[2048, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:285 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[2048, 768]" = torch.ops.prims.inductor_random.default([2048, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[2048, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[2048, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[2048, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, addmm_5);  gt_scalar = addmm_5 = None
        mul_tensor_1: "f16[2048, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:287 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_tensor: "f16[2048, 768]" = torch.ops.aten.add.Tensor(view_12, mul_tensor_1);  view_12 = mul_tensor_1 = None
        reshape_default: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor, [4, 512, 768]);  add_tensor = None
        return reshape_default


def _default_make_inputs():
    return [
    torch.randint(0, 2, [2], dtype=torch.int64, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
