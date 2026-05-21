"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 61199dbd4717
Shape hash: ffd72ad6
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
    def forward(self, tangents_2: "f32[128, 128, 20005]", sub_39: "f32[128, 128, 20005]", primals_200: "f32[20005, 768]", sub_37: "f32[128, 2]", tangents_1: "f32[128, 2]", primals_198: "f32[2, 768]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:61 in forward, code: return self.softmax(self.linear(x))
        sum_dim_int_list: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(tangents_2, [-1], True)
        exp_default: "f32[128, 128, 20005]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        mul_tensor: "f32[128, 128, 20005]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(tangents_2, mul_tensor);  tangents_2 = mul_tensor = None
        reshape_default: "f32[16384, 20005]" = torch.ops.aten.reshape.default(sub_tensor, _shape_param_0);  sub_tensor = _shape_param_0 = None
        permute_default: "f32[768, 20005]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
        permute_default_1: "f32[20005, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        exp_default_1: "f32[128, 2]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_dim_int_list_1: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(tangents_1, [-1], True)
        mul_tensor_1: "f32[128, 2]" = torch.ops.aten.mul.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 2]" = torch.ops.aten.sub.Tensor(tangents_1, mul_tensor_1);  tangents_1 = mul_tensor_1 = None
        permute_default_2: "f32[768, 2]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        permute_default_3: "f32[2, 768]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default, permute_default_1, sub_tensor_1, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([128, 128, 20005], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 20005], dtype=torch.float32, device='cuda'),
    torch.randn([20005, 768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 2], dtype=torch.float32, device='cuda'),
    torch.randn([128, 2], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    [16384, 20005],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
