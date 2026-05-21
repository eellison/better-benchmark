"""
Standalone repro captured via capture_hook.
Label: torchbench_LearningToPaint_train
Pattern hash: c312567199db
Shape hash: 6297a667
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
    def forward(self, mm: "f32[1024, 512]", relu_16: "f32[1024, 512, 4, 4]", convolution_20: "f32[1024, 512, 4, 4]", unsqueeze_86: "f32[1, 512, 1, 1]", squeeze_61: "f32[512]", primals_126: "f32[512]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:134 in forward, code: x = x.view(x.size(0), -1)
        reshape_default: "f32[1024, 512, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:133 in forward, code: x = F.avg_pool2d(x, 4)
        avg_pool2d_backward_default: "f32[1024, 512, 4, 4]" = torch.ops.aten.avg_pool2d_backward.default(reshape_default, relu_16, [4, 4], [], [0, 0], False, True, None);  reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:57 in forward, code: out = F.relu(out)
        le_scalar: "b8[1024, 512, 4, 4]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1024, 512, 4, 4]" = torch.ops.aten.where.self(le_scalar, full_default, avg_pool2d_backward_default);  le_scalar = full_default = avg_pool2d_backward_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:55 in forward, code: out = self.bn2(self.conv2(out))
        sum_dim_int_list: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_86);  convolution_20 = unsqueeze_86 = None
        mul_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.103515625e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.103515625e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_tensor_4: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  squeeze_61 = primals_126 = None
        unsqueeze_default_6: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7


def _default_make_inputs():
    return [
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512, 4, 4], dtype=torch.float32, device='cuda'),
    torch.randn(8388608, dtype=torch.float32, device='cuda').as_strided([1024, 512, 4, 4], [8192, 1, 2048, 512]),  # convolution_20
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    [1024, 512, 1, 1],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
