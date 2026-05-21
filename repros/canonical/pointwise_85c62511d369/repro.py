"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_infer
Pattern hash: 85c62511d369
Shape hash: 72af9b68
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
_shapes_config = "(T([128, 1536, 1, 1], f32), T([128, 1536, 7, 7], f32, stride=(75264, 1, 10752, 1536)), T([128, 1536, 7, 7], f32, stride=(75264, 1, 10752, 1536)))"

class Repro(torch.nn.Module):
    def forward(self, convolution_73: "f32[128, 1536, 1, 1]", convolution_71: "f32[128, 1536, 7, 7]", add_100: "f32[128, 1536, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_default);  convolution_71 = sigmoid_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_1: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_2: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 0.2);  mul_tensor_1 = None
        add_tensor: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, add_100);  mul_tensor_2 = add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_default: "f32[128, 1536, 7, 7]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 1536, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 1536, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 1536, 7, 7]" = torch.ops.aten.div.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None
        mul_tensor_3: "f32[128, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(div_tensor, 0.9622504486493761);  div_tensor = None
        return mul_tensor_3



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
