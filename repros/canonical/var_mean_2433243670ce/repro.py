"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 2433243670ce
Shape hash: d609ab21
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 1536, 1, 1], f32), T([128, 1536, 6, 6], f32, stride=(55296, 1, 9216, 1536)), T([], f32), T([128, 1536, 6, 6], f32, stride=(55296, 1, 9216, 1536)), T([3072, 1536, 1, 1], f32), T([3072, 1, 1, 1], f32), S([1, 3072, -1]), S([3072, 1536, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_79: "f32[128, 1536, 1, 1]", convolution_77: "f32[128, 1536, 6, 6]", primals_229: "f32[]", add_109: "f32[128, 1536, 6, 6]", primals_230: "f32[3072, 1536, 1, 1]", primals_231: "f32[3072, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_default);  convolution_77 = sigmoid_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_1: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_2: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_229);  mul_tensor_1 = primals_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_3: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None
        add_tensor: "f32[128, 1536, 6, 6]" = torch.ops.aten.add.Tensor(mul_tensor_3, add_109);  mul_tensor_3 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 3072, 1536]" = torch.ops.aten.reshape.default(primals_230, _shape_param_0);  primals_230 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_4: "f32[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_231, 0.02551551815399144);  primals_231 = None
        reshape_default_1: "f32[3072]" = torch.ops.aten.reshape.default(mul_tensor_4, [-1]);  mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 3072, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 3072, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 3072, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 3072, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        mul_tensor_5: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[3072, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_6: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_5, unsqueeze_default);  mul_tensor_5 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[3072, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None
        return (add_tensor, reshape_default_2)


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
