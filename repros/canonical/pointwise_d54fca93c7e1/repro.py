"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer
Pattern hash: d54fca93c7e1
Shape hash: d7517139
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
_shapes_config = "()"

class Repro(torch.nn.Module):
    def forward(self):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1484 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full_default: "f32[1, 4096]" = torch.ops.aten.full.default([1, 4096], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:998 in get_extended_attention_mask, code: extended_attention_mask = attention_mask[:, None, None, :]
        unsqueeze_default: "f32[1, 1, 4096]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[1, 1, 1, 4096]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1009 in get_extended_attention_mask, code: extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        convert_element_type_default: "f16[1, 1, 1, 4096]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float16);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1010 in get_extended_attention_mask, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        sub_tensor: "f16[1, 1, 1, 4096]" = torch.ops.aten.sub.Tensor(1.0, convert_element_type_default);  convert_element_type_default = None
        return sub_tensor



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
