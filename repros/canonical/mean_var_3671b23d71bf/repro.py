"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 3671b23d71bf
Shape hash: 22336181
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
    def forward(self, addmm_69: "f32[16384, 768]", inductor_seeds_default: "i64[61]", mul_166: "f32[128, 128, 768]", primals_192: "f32[768]", primals_193: "f32[768]", primals_194: "f32[3072, 768]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        reshape_default: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(addmm_69, _shape_param_0);  addmm_69 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_166, mul_tensor_1);  mul_166 = mul_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_dim: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_tensor, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_correction: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_tensor, [-1], correction = 1.0, keepdim = True)
        sqrt_default: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_correction);  var_correction = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_tensor: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_tensor, mean_dim);  add_tensor = mean_dim = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_192, sub_tensor);  primals_192 = sub_tensor = None
        add_tensor_1: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_default, 1e-06);  sqrt_default = None
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor_2, add_tensor_1);  mul_tensor_2 = add_tensor_1 = None
        add_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_tensor, primals_193);  div_tensor = primals_193 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 61, [61], dtype=torch.int64, device='cuda'),
    torch.randn([128, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    [128, 128, 768],  # _shape_param_0
    [16384, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
