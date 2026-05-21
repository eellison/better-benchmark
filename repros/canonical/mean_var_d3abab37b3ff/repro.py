"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: d3abab37b3ff
Shape hash: 89b7d3a5
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
    def forward(self, primals_2: "f32[20005, 768]", primals_1: "i64[128, 128]", primals_3: "f32[1, 512, 768]", primals_4: "f32[3, 768]", primals_5: "i64[128, 128]", primals_6: "f32[768]", primals_7: "f32[768]", primals_8: "f32[768, 768]", primals_10: "f32[768, 768]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        embedding_default: "f32[128, 128, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 0);  primals_2 = primals_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py:28 in forward, code: return self.pe[:, : x.size(1)]
        slice_tensor: "f32[1, 128, 768]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128);  primals_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(embedding_default, slice_tensor);  embedding_default = slice_tensor = None
        embedding_default_1: "f32[128, 128, 768]" = torch.ops.aten.embedding.default(primals_4, primals_5, 0);  primals_4 = primals_5 = None
        add_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_1);  add_tensor = embedding_default_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[61]" = torch.ops.prims.inductor_seeds.default(61, device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33 in forward, code: return self.dropout(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_1);  gt_scalar = add_tensor_1 = None
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        mean_dim: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_tensor_1, [-1], True)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        var_correction: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_tensor_1, [-1], correction = 1.0, keepdim = True)
        sqrt_default: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_correction);  var_correction = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sub_tensor: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, mean_dim);  mul_tensor_1 = mean_dim = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_6, sub_tensor);  primals_6 = sub_tensor = None
        add_tensor_2: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_default, 1e-06);  sqrt_default = None
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor_2, add_tensor_2);  mul_tensor_2 = add_tensor_2 = None
        add_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_tensor, primals_7);  div_tensor = primals_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_0);  add_tensor_3 = _shape_param_0 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        return (reshape_default, permute_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([20005, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 20005, [128, 128], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 3, [128, 128], dtype=torch.int64, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [16384, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
