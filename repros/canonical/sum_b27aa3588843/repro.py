"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: b27aa3588843
Shape hash: 102f3a82
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
_shapes_config = "(T([128, 1536, 6, 6], f32, stride=(55296, 1, 9216, 1536)), T([128, 1536, 1, 1], f32), T([], f32), T([128, 1536, 6, 6], f32, stride=(55296, 1, 9216, 1536)))"

class Repro(torch.nn.Module):
    def forward(self, getitem_114: "f32[128, 1536, 6, 6]", convolution_79: "f32[128, 1536, 1, 1]", primals_229: "f32[]", convolution_77: "f32[128, 1536, 6, 6]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(getitem_114, 0.2);  getitem_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_1: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_229);  mul_tensor = primals_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_2: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 2.0);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_3: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_2, convolution_77);  mul_tensor_2 = convolution_77 = None
        sum_dim_int_list: "f32[128, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2, 3], True);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_tensor: "f32[128, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_4: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor);  sigmoid_default = sub_tensor = None
        mul_tensor_5: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_4);  sum_dim_int_list = mul_tensor_4 = None
        return mul_tensor_5



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
