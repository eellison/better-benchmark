"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_training
Pattern hash: d8415e06c1aa
Shape hash: fea081e2
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_57: "f32[4096, 32000]", view_as_real_23: "f32[8, 512, 768, 2]", mul_446: "f32[8, 512, 768]", gt: "b8[8, 512, 768]", _shape_param_0, primals_9: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:331 in forward, code: hidden_states = self.decoder(hidden_states)
        permute_default: "f32[32000, 4096]" = torch.ops.aten.permute.default(view_57, [1, 0]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        select_int: "f32[8, 512, 768]" = torch.ops.aten.select.int(view_as_real_23, 3, 0);  view_as_real_23 = None
        add_tensor: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_446, select_int);  mul_446 = select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:138 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[8, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor);  add_tensor = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:137 in forward, code: embeddings = self.projection(embeddings)
        reshape_default: "f32[4096, 768]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (permute_default, reshape_default, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([4096, 32000], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768, 2], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 512, 768], dtype=torch.bool, device='cuda'),
    [4096, 768],  # _shape_param_0
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
