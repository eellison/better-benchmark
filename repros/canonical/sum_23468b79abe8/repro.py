"""
Standalone repro captured via capture_hook.
Label: torchbench_resnet18_train
Pattern hash: 23468b79abe8
Shape hash: 55ce5ecf
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
    def forward(self, tangents_1: "f32[512, 1000]", sum_3: "f32[512]", squeeze_58: "f32[512]", sum_5: "f32[512]", squeeze_55: "f32[512]", sum_7: "f32[512]", squeeze_52: "f32[512]", sum_9: "f32[512]", squeeze_49: "f32[512]", sum_11: "f32[512]", squeeze_46: "f32[512]", sum_13: "f32[256]", squeeze_43: "f32[256]", sum_15: "f32[256]", squeeze_40: "f32[256]", sum_17: "f32[256]", squeeze_37: "f32[256]", sum_19: "f32[256]", squeeze_34: "f32[256]", sum_21: "f32[256]", squeeze_31: "f32[256]", sum_23: "f32[128]", squeeze_28: "f32[128]", sum_25: "f32[128]", squeeze_25: "f32[128]", sum_27: "f32[128]", squeeze_22: "f32[128]", sum_29: "f32[128]", squeeze_19: "f32[128]", sum_31: "f32[128]", squeeze_16: "f32[128]", sum_33: "f32[64]", squeeze_13: "f32[64]", sum_35: "f32[64]", squeeze_10: "f32[64]", sum_37: "f32[64]", squeeze_7: "f32[64]", sum_39: "f32[64]", squeeze_4: "f32[64]", sum_41: "f32[64]", squeeze_1: "f32[64]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute_default: "f32[1000, 512]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_tensor: "f32[512]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_58);  sum_3 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_tensor_1: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_55);  sum_5 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        mul_tensor_2: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_52);  sum_7 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_tensor_3: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_49);  sum_9 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_tensor_4: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_46);  sum_11 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_tensor_5: "f32[256]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_43);  sum_13 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_tensor_6: "f32[256]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_40);  sum_15 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        mul_tensor_7: "f32[256]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_37);  sum_17 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_tensor_8: "f32[256]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_34);  sum_19 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_tensor_9: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_31);  sum_21 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_tensor_10: "f32[128]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_28);  sum_23 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_tensor_11: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_25);  sum_25 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        mul_tensor_12: "f32[128]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_22);  sum_27 = squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_tensor_13: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_19);  sum_29 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_tensor_14: "f32[128]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_16);  sum_31 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_tensor_15: "f32[64]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_13);  sum_33 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_tensor_16: "f32[64]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_10);  sum_35 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_tensor_17: "f32[64]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_7);  sum_37 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        mul_tensor_18: "f32[64]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_4);  sum_39 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        mul_tensor_19: "f32[64]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_1);  sum_41 = squeeze_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19)


def _default_make_inputs():
    return [
    torch.randn([512, 1000], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
