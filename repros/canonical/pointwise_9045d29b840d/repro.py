"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_2: "f32[6144, 64, 64]", getitem_1: "i64[8, 12, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:930 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        reshape_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_2, [8, 12, 64, 64, 64]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:938 in _attend, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        reshape_default_1: "f32[8, 12, 4096, 64]" = torch.ops.aten.reshape.default(reshape_default, [8, 12, 4096, 64]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:805 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        empty_memory_format: "i64[8, 12, 4096]" = torch.ops.aten.empty.memory_format([8, 12, 4096], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:799 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
        iota_default: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:800 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .view(1, 1, -1)
        reshape_default_2: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_default, [1, 1, -1]);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:801 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .expand(sorted_bucket_idx.shape)
        expand_default: "i64[8, 12, 4096]" = torch.ops.aten.expand.default(reshape_default_2, [8, 12, 4096]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:806 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        scatter_src: "i64[8, 12, 4096]" = torch.ops.aten.scatter.src(empty_memory_format, -1, getitem_1, expand_default);  empty_memory_format = getitem_1 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1114 in forward, code: expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        unsqueeze_default: "i64[8, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(scatter_src, -1);  scatter_src = None
        expand_default_1: "i64[8, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, [8, 12, 4096, 64]);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1115 in forward, code: out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        gather_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.gather.default(reshape_default_1, 2, expand_default_1);  reshape_default_1 = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:413 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_default: "f32[8, 4096, 12, 64]" = torch.ops.aten.permute.default(gather_default, [0, 2, 1, 3]);  gather_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:414 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_default: "f32[8, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(clone_default, [8, 4096, 768]);  clone_default = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([6144, 64, 64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 12, 4096], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
