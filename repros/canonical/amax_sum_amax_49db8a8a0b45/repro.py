"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer
Pattern hash: 49db8a8a0b45
Shape hash: 3b073e13
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
    def forward(self, addmm_72: "f32[128, 2]", addmm_73: "f32[16384, 20005]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        amax_default: "f32[128, 1]" = torch.ops.aten.amax.default(addmm_72, [-1], True)
        sub_tensor: "f32[128, 2]" = torch.ops.aten.sub.Tensor(addmm_72, amax_default);  addmm_72 = amax_default = None
        exp_default: "f32[128, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[128, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[128, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:61 in forward, code: return self.softmax(self.linear(x))
        reshape_default: "f32[128, 128, 20005]" = torch.ops.aten.reshape.default(addmm_73, _shape_param_0);  addmm_73 = _shape_param_0 = None
        amax_default_1: "f32[128, 128, 1]" = torch.ops.aten.amax.default(reshape_default, [-1], True)
        sub_tensor_2: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(reshape_default, amax_default_1);  reshape_default = amax_default_1 = None
        exp_default_1: "f32[128, 128, 20005]" = torch.ops.aten.exp.default(sub_tensor_2)
        sum_dim_int_list_1: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [-1], True);  exp_default_1 = None
        log_default_1: "f32[128, 128, 1]" = torch.ops.aten.log.default(sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_3: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(sub_tensor_2, log_default_1);  sub_tensor_2 = log_default_1 = None
        return (sub_tensor_1, sub_tensor_3)


def _default_make_inputs():
    return [
    torch.randn([128, 2], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 20005], dtype=torch.float32, device='cuda'),
    [128, 128, 20005],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
