"""
Standalone repro captured via capture_hook.
Label: torchbench_LearningToPaint_train
Pattern hash: eaeef99443dd
Shape hash: b4968b7b
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
    def forward(self, mul_148: "f32[1024, 65]", sum_3: "f32[512]", squeeze_61: "f32[512]", sum_5: "f32[512]", squeeze_58: "f32[512]", sum_7: "f32[512]", squeeze_55: "f32[512]", sum_9: "f32[512]", squeeze_52: "f32[512]", sum_11: "f32[512]", squeeze_49: "f32[512]", sum_13: "f32[256]", squeeze_46: "f32[256]", sum_15: "f32[256]", squeeze_43: "f32[256]", sum_17: "f32[256]", squeeze_40: "f32[256]", sum_19: "f32[256]", squeeze_37: "f32[256]", sum_21: "f32[256]", squeeze_34: "f32[256]", sum_23: "f32[128]", squeeze_31: "f32[128]", sum_25: "f32[128]", squeeze_28: "f32[128]", sum_27: "f32[128]", squeeze_25: "f32[128]", sum_29: "f32[128]", squeeze_22: "f32[128]", sum_31: "f32[128]", squeeze_19: "f32[128]", sum_33: "f32[64]", squeeze_16: "f32[64]", sum_35: "f32[64]", squeeze_13: "f32[64]", sum_37: "f32[64]", squeeze_10: "f32[64]", sum_39: "f32[64]", squeeze_7: "f32[64]", sum_41: "f32[64]", squeeze_4: "f32[64]", sum_43: "f32[64]", squeeze_1: "f32[64]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:135 in forward, code: x = self.fc(x)
        permute_default: "f32[65, 1024]" = torch.ops.aten.permute.default(mul_148, [1, 0])
        sum_dim_int_list: "f32[1, 65]" = torch.ops.aten.sum.dim_IntList(mul_148, [0], True);  mul_148 = None
        reshape_default: "f32[65]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        mul_tensor: "f32[512]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_61);  sum_3 = squeeze_61 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        mul_tensor_1: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_58);  sum_5 = squeeze_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        mul_tensor_2: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_55);  sum_7 = squeeze_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        mul_tensor_3: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_52);  sum_9 = squeeze_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        mul_tensor_4: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_49);  sum_11 = squeeze_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        mul_tensor_5: "f32[256]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_46);  sum_13 = squeeze_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        mul_tensor_6: "f32[256]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_43);  sum_15 = squeeze_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        mul_tensor_7: "f32[256]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_40);  sum_17 = squeeze_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        mul_tensor_8: "f32[256]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_37);  sum_19 = squeeze_37 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        mul_tensor_9: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_34);  sum_21 = squeeze_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        mul_tensor_10: "f32[128]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_31);  sum_23 = squeeze_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        mul_tensor_11: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_28);  sum_25 = squeeze_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        mul_tensor_12: "f32[128]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_25);  sum_27 = squeeze_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        mul_tensor_13: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_22);  sum_29 = squeeze_22 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        mul_tensor_14: "f32[128]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_19);  sum_31 = squeeze_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        mul_tensor_15: "f32[64]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_16);  sum_33 = squeeze_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        mul_tensor_16: "f32[64]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_13);  sum_35 = squeeze_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:56 in forward, code: out += self.shortcut(x)
        mul_tensor_17: "f32[64]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_10);  sum_37 = squeeze_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        mul_tensor_18: "f32[64]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_7);  sum_39 = squeeze_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:54 in forward, code: out = F.relu(self.bn1(self.conv1(x)))
        mul_tensor_19: "f32[64]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_4);  sum_41 = squeeze_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:128 in forward, code: x = F.relu(self.bn1(self.conv1(x)))
        mul_tensor_20: "f32[64]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_1);  sum_43 = squeeze_1 = None
        return (permute_default, reshape_default, mul_tensor, mul_tensor_1, mul_tensor_2, mul_tensor_3, mul_tensor_4, mul_tensor_5, mul_tensor_6, mul_tensor_7, mul_tensor_8, mul_tensor_9, mul_tensor_10, mul_tensor_11, mul_tensor_12, mul_tensor_13, mul_tensor_14, mul_tensor_15, mul_tensor_16, mul_tensor_17, mul_tensor_18, mul_tensor_19, mul_tensor_20)


def _default_make_inputs():
    return [
    torch.randn([1024, 65], dtype=torch.float32, device='cuda'),
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
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    [65],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
