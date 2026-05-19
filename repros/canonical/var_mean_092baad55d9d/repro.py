"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: 092baad55d9d
Shape hash: 838bcbcb
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_38: "f32[128, 320, 14, 14]", add_71: "f32[128, 320, 14, 14]", primals_134: "f32[320]", primals_135: "f32[320]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[128, 320, 14, 14]" = torch.ops.aten.add.Tensor(convolution_38, add_71);  convolution_38 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default: "f32[128, 14, 14, 320]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 3, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 14, 14, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 14, 14, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 14, 14, 320]" = torch.ops.aten.sub.Tensor(permute_default, getitem_1);  permute_default = getitem_1 = None
        mul_tensor: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 14, 14, 320]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_134);  mul_tensor = primals_134 = None
        add_tensor_2: "f32[128, 14, 14, 320]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_135);  mul_tensor_1 = primals_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default_1: "f32[128, 320, 14, 14]" = torch.ops.aten.permute.default(add_tensor_2, [0, 3, 1, 2]);  add_tensor_2 = None
        return permute_default_1


def _default_make_inputs():
    return [
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 4480, 320]),  # convolution_38
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([128, 320, 14, 14], [62720, 1, 4480, 320]),  # add_71
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
