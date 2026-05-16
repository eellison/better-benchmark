"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_training
Pattern hash: b895718d7c0f
Shape hash: 5b5352d3
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[8, 512, 1]", add_1: "f32[8, 512, 768]", getitem_1: "f32[8, 512, 1]", primals_7: "f32[768]", primals_8: "f32[768]", _shape_param_0, primals_9: "f32[384, 768]", primals_16: "f32[384, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:104 in forward, code: embeddings = self.LayerNorm(embeddings)
        add_tensor: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_tensor: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_7);  mul_tensor = primals_7 = None
        add_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_8);  mul_tensor_1 = primals_8 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[37]" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:105 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 512, 768]" = torch.ops.prims.inductor_random.default([8, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_1);  gt_scalar = add_tensor_1 = None
        mul_tensor_3: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        reshape_default: "f32[4096, 768]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[768, 384]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_default_1: "f32[8, 768, 512]" = torch.ops.aten.permute.default(mul_tensor_3, [0, 2, 1]);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        permute_default_2: "f32[768, 384]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        return (reshape_default, permute_default, permute_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [4096, 768],  # _shape_param_0
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([384, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
