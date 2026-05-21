"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: e00c54d5e104
Shape hash: 51e196c5
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
    def forward(self, mm_6: "f32[16384, 768]", primals_192: "f32[768]", sub_35: "f32[128, 128, 768]", sqrt_23: "f32[128, 128, 1]", mul_185: "f32[128, 128, 768]", gt_58: "b8[128, 128, 768]", primals_190: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        reshape_default: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_6, _shape_param_0);  mm_6 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_192, sub_35)
        add_tensor: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_23, 1e-06)
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor, add_tensor);  mul_tensor = None
        div_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_tensor, add_tensor);  div_tensor = None
        neg_default: "f32[128, 128, 768]" = torch.ops.aten.neg.default(reshape_default)
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None
        sum_dim_int_list: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, primals_192);  div_tensor_2 = primals_192 = None
        neg_default_1: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_tensor_2)
        sum_dim_int_list_1: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_default_1, [2], True);  neg_default_1 = None
        add_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_185, mul_tensor_2);  mul_185 = mul_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_scalar: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt_23, 2)
        div_tensor_3: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_dim_int_list, mul_scalar);  sum_dim_int_list = mul_scalar = None
        eq_scalar: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt_23, 0);  sqrt_23 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_scalar, full_default, div_tensor_3);  eq_scalar = full_default = div_tensor_3 = None
        mul_scalar_1: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_self, 0.002607561929595828);  where_self = None
        mul_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_scalar_1, sub_35);  mul_scalar_1 = sub_35 = None
        add_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_3);  add_tensor_1 = mul_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_default: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        div_scalar: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        add_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, div_scalar);  add_tensor_2 = div_scalar = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_default: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_58, torch.float32);  gt_58 = None
        mul_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_4);  add_tensor_3 = mul_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_2);  mul_tensor_5 = _shape_param_2 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [128, 128, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [128, 128, 768],  # _shape_param_0
    [128, 128, 768],  # _shape_param_1
    [16384, 768],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
