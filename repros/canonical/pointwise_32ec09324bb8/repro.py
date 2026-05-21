"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: 32ec09324bb8
Shape hash: d4c044fc
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
_shapes_config = "(T([768, 64, 64], f16), T([1, 12, 4096], i64, gen=Index(4096)), S([1, 12, 64, 64, 64]), S([1, 12, 4096, 64]), S([1, 12, 4096]), S([1, 12, 4096, 64]), S([1, 4096, 768]), S([4096, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_14: "f16[768, 64, 64]", getitem_29: "i64[1, 12, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        reshape_default: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_14, _shape_param_0);  bmm_14 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:904 in _attend, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        reshape_default_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:776 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        empty_memory_format: "i64[1, 12, 4096]" = torch.ops.aten.empty.memory_format([1, 12, 4096], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:770 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
        iota_default: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:771 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .view(1, 1, -1)
        reshape_default_2: "i64[1, 1, 4096]" = torch.ops.aten.reshape.default(iota_default, [1, 1, -1]);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:772 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: .expand(sorted_bucket_idx.shape)
        expand_default: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(reshape_default_2, _shape_param_2);  reshape_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:777 in _get_sorted_bucket_idx_and_undo_sorted_bucket_idx, code: undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        scatter_src: "i64[1, 12, 4096]" = torch.ops.aten.scatter.src(empty_memory_format, -1, getitem_29, expand_default);  empty_memory_format = getitem_29 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1080 in forward, code: expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        unsqueeze_default: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(scatter_src, -1);  scatter_src = None
        expand_default_1: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_3);  unsqueeze_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1081 in forward, code: out_vectors = torch.gather(out_vectors, 2, expanded_undo_sort_indices)
        gather_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(reshape_default_1, 2, expand_default_1);  reshape_default_1 = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:387 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(gather_default, [0, 2, 1, 3]);  gather_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:388 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1331 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_4: "f16[4096, 768]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None
        return reshape_default_4



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
