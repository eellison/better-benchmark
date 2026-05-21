"""
Standalone repro captured via capture_hook.
Label: torchbench_nanogpt_train
Pattern hash: 6834871280ca
Shape hash: b3639ad3
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([50304, 768], f32), T([1, 64], i64, gen=Index(50304)), T([1024, 768], f32), T([768], f32), T([768], f32), S([64, 768]))"

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[50304, 768]", primals_1: "i64[1, 64]", primals_3: "f32[1024, 768]", primals_4: "f32[768]", primals_5: "f32[768]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:224 in forward, code: pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
        iota_default: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 64]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:229 in forward, code: tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        embedding_default: "f32[1, 64, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = primals_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:230 in forward, code: pos_emb = self.transformer.wpe(
        embedding_default_1: "f32[1, 64, 768]" = torch.ops.aten.embedding.default(primals_3, unsqueeze_default);  primals_3 = unsqueeze_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:233 in forward, code: x = self.transformer.drop(tok_emb + pos_emb)
        add_tensor: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_4);  mul_tensor = primals_4 = None
        add_tensor_2: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_5);  mul_tensor_1 = primals_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:83 in forward, code: q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        reshape_default: "f32[64, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_0);  add_tensor_2 = _shape_param_0 = None
        return reshape_default



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
