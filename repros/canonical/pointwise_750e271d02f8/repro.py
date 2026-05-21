"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: 750e271d02f8
Shape hash: 669a204e
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
_shapes_config = "(T([1, 6, 6144, 128], bf16, stride=(4718592, 128, 768, 1)), T([6144, 6], bf16), S([1, 6144, 6, 1]), S([1, 6144, 768]), S([6144, 768]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_119: "bf16[1, 6, 6144, 128]", mm_default: "bf16[6144, 6]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:696 in forward, code: ).transpose(1, 2)
        permute_default: "bf16[1, 6144, 6, 128]" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        unsqueeze_default: "bf16[1, 6144, 6]" = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        sigmoid_default: "bf16[1, 6144, 6]" = torch.ops.aten.sigmoid.default(unsqueeze_default);  unsqueeze_default = None
        reshape_default: "bf16[1, 6144, 6, 1]" = torch.ops.aten.reshape.default(sigmoid_default, _shape_param_0);  sigmoid_default = _shape_param_0 = None
        mul_tensor: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_default, reshape_default);  permute_default = reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:701 in forward, code: y = y.contiguous().view(
        reshape_default_1: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:704 in forward, code: y = F.linear(y, self.qkvo_w[3].type_as(y))
        reshape_default_2: "bf16[6144, 768]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        return reshape_default_2



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
