"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_training
Pattern hash: 39aa3f76778b
Shape hash: 91e2246e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[320, 256]", primals_1: "i64[8, 4096]", primals_3: "f32[64, 1, 64]", _shape_param_0, primals_4: "f32[1, 64, 192]", _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:330 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[8, 4096, 256]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = primals_1 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:339 in forward, code: embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default: "f32[8, 4096, 256]" = torch.ops.prims.inductor_random.default([8, 4096, 256], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 4096, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None
        mul_tensor: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(gt_scalar, embedding_default);  gt_scalar = embedding_default = None
        mul_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.0526315789473684);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:227 in forward, code: weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        expand_default: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(primals_3, _shape_param_0);  primals_3 = _shape_param_0 = None
        expand_default_1: "f32[8, 64, 64, 192]" = torch.ops.aten.expand.default(primals_4, _shape_param_1);  primals_4 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:240 in forward, code: weights = torch.cat(broadcasted_weights, dim=-1)
        cat_default: "f32[8, 64, 64, 256]" = torch.ops.aten.cat.default([expand_default, expand_default_1], -1);  expand_default = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:242 in forward, code: transposed_weights = weights.transpose(2, 1)
        permute_default: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(cat_default, [0, 2, 1, 3]);  cat_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:244 in forward, code: dropped_transposed_weights = nn.functional.dropout2d(
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default_1: "f32[8, 64, 1, 1]" = torch.ops.prims.inductor_random.default([8, 64, 1, 1], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        lt_scalar: "b8[8, 64, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default_1, 0.95);  inductor_random_default_1 = None
        convert_element_type_default: "f32[8, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None
        div_scalar: "f32[8, 64, 1, 1]" = torch.ops.aten.div.Scalar(convert_element_type_default, 0.95);  convert_element_type_default = None
        mul_tensor_2: "f32[8, 64, 64, 256]" = torch.ops.aten.mul.Tensor(permute_default, div_scalar);  permute_default = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:247 in forward, code: dropped_weights = dropped_transposed_weights.transpose(2, 1)
        permute_default_1: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(mul_tensor_2, [0, 2, 1, 3]);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:249 in forward, code: position_encodings = torch.reshape(dropped_weights, (batch_size, sequence_length, -1))
        reshape_default: "f32[8, 4096, 256]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:343 in forward, code: embeddings = embeddings + position_embeddings
        add_tensor: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(mul_tensor_1, reshape_default);  mul_tensor_1 = reshape_default = None
        return add_tensor


def _default_make_inputs():
    return [
    torch.randn([320, 256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 4096], dtype=torch.int64, device='cuda'),
    torch.randn([64, 1, 64], dtype=torch.float32, device='cuda'),
    [8, 64, 64, 64],  # _shape_param_0
    torch.randn([1, 64, 192], dtype=torch.float32, device='cuda'),
    [8, 64, 64, 192],  # _shape_param_1
    [8, 4096, -1],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
