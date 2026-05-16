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
    def forward(self, add_261: "f32[512, 16, 1024]", getitem_95: "f32[512, 16, 1]", getitem_94: "f32[512, 16, 1]", arg360_1: "f32[1024]", arg361_1: "f32[1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:468 in forward, code: output = self.layer_norm(output + inp)
        sub_tensor: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_261, getitem_95);  add_261 = getitem_95 = None
        add_tensor: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_default: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg360_1);  mul_tensor = arg360_1 = None
        add_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg361_1);  mul_tensor_1 = arg361_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1391 in forward, code: output = output.permute(1, 0, 2).contiguous()
        permute_default: "f32[16, 512, 1024]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        clone_default: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1633 in forward, code: logits = self.lm_loss(transformer_outputs[0])
        reshape_default: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_default, [8192, 1024]);  clone_default = None
        return reshape_default


def _default_make_inputs():
    return [
    torch.randn([512, 16, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
