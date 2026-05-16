"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_162: "f32[8, 1024, 768]", getitem_43: "f32[8, 1024, 1]", getitem_42: "f32[8, 1024, 1]", arg177_1: "f32[768]", arg178_1: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1175 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_162, getitem_43);  add_162 = getitem_43 = None
        add_tensor: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg177_1);  mul_tensor = arg177_1 = None
        add_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg178_1);  mul_tensor_1 = arg178_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_default: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:508 in forward, code: query_vectors = self.query(hidden_states)
        clone_default: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format)
        reshape_default: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_default, [8192, 768]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:509 in forward, code: key_vectors = self.key(hidden_states)
        clone_default_1: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format)
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_default_1, [8192, 768]);  clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:510 in forward, code: value_vectors = self.value(hidden_states)
        clone_default_2: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_default_2, [8192, 768]);  clone_default_2 = None
        return (reshape_default, reshape_default_1, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
