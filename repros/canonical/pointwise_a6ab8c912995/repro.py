"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_inference
Pattern hash: a6ab8c912995
Shape hash: d440f5eb
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, _tensor_constant0: "i64[]", bmm_2: "f32[6144, 64, 64]", _shape_param_0, _shape_param_1, _shape_param_2, getitem_1: "i64[8, 12, 4096]", _shape_param_3, _shape_param_4, view_5: "i64[8, 12, 4096]", _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:598 in torch_dynamo_resume_in_forward_at_544, code: sqrt_num = np.sqrt(self.attention_head_size)
        lift_fresh_copy_default: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        convert_element_type_default: "f64[]" = torch.ops.prims.convert_element_type.default(lift_fresh_copy_default, torch.float64);  lift_fresh_copy_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        reshape_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_2, _shape_param_0);  bmm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:904 in _attend, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        reshape_default_1: "f32[8, 12, 4096, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:776 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        empty_memory_format: "i64[8, 12, 4096]" = torch.ops.aten.empty.memory_format([8, 12, 4096], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:770 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
        iota_default: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:771 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .view(1, 1, -1)
        reshape_default_2: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_default, [1, 1, -1]);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:772 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .expand(sorted_bucket_idx.shape)
        expand_default: "i64[8, 12, 4096]" = torch.ops.aten.expand.default(reshape_default_2, _shape_param_2);  reshape_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:777 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        scatter_src: "i64[8, 12, 4096]" = torch.ops.aten.scatter.src(empty_memory_format, -1, getitem_1, expand_default);  empty_memory_format = getitem_1 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1080 in forward, code: expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        unsqueeze_default: "i64[8, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(scatter_src, -1);  scatter_src = None
        expand_default_1: "i64[8, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_3);  unsqueeze_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1081 in forward, code: out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        gather_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.gather.default(reshape_default_1, 2, expand_default_1);  reshape_default_1 = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_default: "f32[8, 4096, 12, 64]" = torch.ops.aten.permute.default(gather_default, [0, 2, 1, 3]);  gather_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_default: "f32[8, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:674 in torch_dynamo_resume_in_forward_at_544, code: self.num_attention_heads, self.attention_head_size, self.hidden_size
        reshape_default_4: "i64[8, 12, 1, 4096]" = torch.ops.aten.reshape.default(view_5, _shape_param_5);  view_5 = _shape_param_5 = None
        return (convert_element_type_default, reshape_default_3, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randn([6144, 64, 64], dtype=torch.float32, device='cuda'),
    [8, 12, 64, 64, 64],  # _shape_param_0
    [8, 12, 4096, 64],  # _shape_param_1
    [8, 12, 4096],  # _shape_param_2
    torch.randint(0, 2, [8, 12, 4096], dtype=torch.int64, device='cuda'),
    [8, 12, 4096, 64],  # _shape_param_3
    [8, 4096, 768],  # _shape_param_4
    torch.randint(0, 2, [8, 12, 4096], dtype=torch.int64, device='cuda'),
    [8, 12, 1, -1],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
