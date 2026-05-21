"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train
Pattern hash: a223aa214255
Shape hash: 3c015f72
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
_shapes_config = "(T([2, 1024], i64), T([2, 1024], i32), T([50265, 768], f32), T([2, 1024], i64, gen=Index(50265)), T([4098, 768], f32), T([1, 768], f32), T([768], f32), T([768], f32))"

class Repro(torch.nn.Module):
    def forward(self, cumsum: "i64[2, 1024]", convert_element_type: "i32[2, 1024]", primals_2: "f32[50265, 768]", primals_1: "i64[2, 1024]", primals_3: "f32[4098, 768]", primals_4: "f32[1, 768]", primals_5: "f32[768]", primals_6: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1486 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:379 in create_position_ids_from_input_ids, code: incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        convert_element_type_default: "i32[2, 1024]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        mul_tensor: "i32[2, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type);  convert_element_type_default = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:380 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_default_1: "i64[2, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor: "i64[2, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:418 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[2, 1024, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 1);  primals_2 = primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:419 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_1: "f32[2, 1024, 768]" = torch.ops.aten.embedding.default(primals_3, add_tensor, 1);  primals_3 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:420 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_2: "f32[2, 1024, 768]" = torch.ops.aten.embedding.default(primals_4, full_default);  primals_4 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:422 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_tensor_1: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        add_tensor_2: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default_2);  add_tensor_1 = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:423 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[2, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[2, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_3: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        sub_tensor: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1);  add_tensor_2 = getitem_1 = None
        mul_tensor_1: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_5);  mul_tensor_1 = primals_5 = None
        add_tensor_4: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, primals_6);  mul_tensor_2 = primals_6 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:424 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[2, 1024, 768]" = torch.ops.prims.inductor_random.default([2, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[2, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_3: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_4);  gt_scalar = add_tensor_4 = None
        mul_tensor_4: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:423 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_tensor: "f32[2, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        return (div_tensor, mul_tensor_4)



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
