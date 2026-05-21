"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train
Pattern hash: 19fb2b130301
Shape hash: c87a98c6
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
    def forward(self, tangents_1: "f32[8, 2, 640, 959]", sum_3: "f32[64]", squeeze_52: "f32[64]", mul_153: "f32[8, 64, 640, 959]", sum_6: "f32[64]", squeeze_49: "f32[64]", mul_162: "f32[8, 64, 640, 959]", sum_9: "f32[64]", squeeze_46: "f32[64]", mul_174: "f32[8, 64, 320, 479]", sum_12: "f32[128]", squeeze_43: "f32[128]", mul_183: "f32[8, 128, 320, 479]", sum_15: "f32[128]", squeeze_40: "f32[128]", mul_195: "f32[8, 128, 160, 239]", sum_18: "f32[256]", squeeze_37: "f32[256]", mul_204: "f32[8, 256, 160, 239]", sum_21: "f32[256]", squeeze_34: "f32[256]", mul_216: "f32[8, 256, 80, 119]", sum_24: "f32[512]", squeeze_31: "f32[512]", mul_225: "f32[8, 512, 80, 119]", sum_27: "f32[512]", squeeze_28: "f32[512]", mul_237: "f32[8, 512, 40, 59]", sum_30: "f32[512]", squeeze_25: "f32[512]", mul_246: "f32[8, 512, 40, 59]", sum_33: "f32[512]", squeeze_22: "f32[512]", mul_255: "f32[8, 512, 80, 119]", sum_36: "f32[512]", squeeze_19: "f32[512]", mul_264: "f32[8, 512, 80, 119]", sum_39: "f32[256]", squeeze_16: "f32[256]", mul_273: "f32[8, 256, 160, 239]", sum_42: "f32[256]", squeeze_13: "f32[256]", mul_282: "f32[8, 256, 160, 239]", sum_45: "f32[128]", squeeze_10: "f32[128]", mul_291: "f32[8, 128, 320, 479]", sum_48: "f32[128]", squeeze_7: "f32[128]", mul_300: "f32[8, 128, 320, 479]", sum_51: "f32[64]", squeeze_4: "f32[64]", mul_309: "f32[8, 64, 640, 959]", sum_54: "f32[64]", squeeze_1: "f32[64]", mul_318: "f32[8, 64, 640, 959]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:77 in forward, code: return self.conv(x)
        sum_dim_int_list: "f32[2]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 2, 3]);  tangents_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_52);  sum_3 = squeeze_52 = None
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 2, 3]);  mul_153 = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_49);  sum_6 = squeeze_49 = None
        sum_dim_int_list_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_162, [0, 2, 3]);  mul_162 = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_46);  sum_9 = squeeze_46 = None
        sum_dim_int_list_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 2, 3]);  mul_174 = None
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_43);  sum_12 = squeeze_43 = None
        sum_dim_int_list_4: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_183, [0, 2, 3]);  mul_183 = None
        mul_tensor_4: "f32[128]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_40);  sum_15 = squeeze_40 = None
        sum_dim_int_list_5: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 2, 3]);  mul_195 = None
        mul_tensor_5: "f32[256]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_37);  sum_18 = squeeze_37 = None
        sum_dim_int_list_6: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3]);  mul_204 = None
        mul_tensor_6: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_34);  sum_21 = squeeze_34 = None
        sum_dim_int_list_7: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 2, 3]);  mul_216 = None
        mul_tensor_7: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_31);  sum_24 = squeeze_31 = None
        sum_dim_int_list_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 2, 3]);  mul_225 = None
        mul_tensor_8: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_28);  sum_27 = squeeze_28 = None
        sum_dim_int_list_9: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 2, 3]);  mul_237 = None
        mul_tensor_9: "f32[512]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_25);  sum_30 = squeeze_25 = None
        sum_dim_int_list_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 2, 3]);  mul_246 = None
        mul_tensor_10: "f32[512]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_22);  sum_33 = squeeze_22 = None
        sum_dim_int_list_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 2, 3]);  mul_255 = None
        mul_tensor_11: "f32[512]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_19);  sum_36 = squeeze_19 = None
        sum_dim_int_list_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 2, 3]);  mul_264 = None
        mul_tensor_12: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_16);  sum_39 = squeeze_16 = None
        sum_dim_int_list_13: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 2, 3]);  mul_273 = None
        mul_tensor_13: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_13);  sum_42 = squeeze_13 = None
        sum_dim_int_list_14: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
        mul_tensor_14: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_10);  sum_45 = squeeze_10 = None
        sum_dim_int_list_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 2, 3]);  mul_291 = None
        mul_tensor_15: "f32[128]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_7);  sum_48 = squeeze_7 = None
        sum_dim_int_list_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 2, 3]);  mul_300 = None
        mul_tensor_16: "f32[64]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_4);  sum_51 = squeeze_4 = None
        sum_dim_int_list_17: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 2, 3]);  mul_309 = None
        mul_tensor_17: "f32[64]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_1);  sum_54 = squeeze_1 = None
        sum_dim_int_list_18: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 2, 3]);  mul_318 = None
        return (sum_dim_int_list, mul_tensor, sum_dim_int_list_1, mul_tensor_1, sum_dim_int_list_2, mul_tensor_2, sum_dim_int_list_3, mul_tensor_3, sum_dim_int_list_4, mul_tensor_4, sum_dim_int_list_5, mul_tensor_5, sum_dim_int_list_6, mul_tensor_6, sum_dim_int_list_7, mul_tensor_7, sum_dim_int_list_8, mul_tensor_8, sum_dim_int_list_9, mul_tensor_9, sum_dim_int_list_10, mul_tensor_10, sum_dim_int_list_11, mul_tensor_11, sum_dim_int_list_12, mul_tensor_12, sum_dim_int_list_13, mul_tensor_13, sum_dim_int_list_14, mul_tensor_14, sum_dim_int_list_15, mul_tensor_15, sum_dim_int_list_16, mul_tensor_16, sum_dim_int_list_17, mul_tensor_17, sum_dim_int_list_18)


def _default_make_inputs():
    return [
    torch.randn([8, 2, 640, 959], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([8, 64, 640, 959], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([8, 64, 640, 959], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([8, 64, 320, 479], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 320, 479], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 160, 239], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([8, 256, 160, 239], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([8, 256, 80, 119], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 80, 119], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 40, 59], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 40, 59], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 80, 119], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 80, 119], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([8, 256, 160, 239], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([8, 256, 160, 239], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 320, 479], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 320, 479], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([8, 64, 640, 959], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([8, 64, 640, 959], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
