import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:64 in shift_tokens_right, code: shifted_input_ids[:, 0] = decoder_start_token_id
        _tensor_constant0: "i64[]" = self._tensor_constant0;  _tensor_constant0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:62 in shift_tokens_right, code: shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        full: "i64[4, 512]" = torch.ops.aten.full.default([4, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:63 in shift_tokens_right, code: shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        slice_2: "i64[4, 511]" = torch.ops.aten.slice.Tensor(full, 1, 1, 9223372036854775807)
        slice_1: "i64[4, 511]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, -1);  arg0_1 = None
        clone: "i64[4, 511]" = torch.ops.aten.clone.default(slice_1);  slice_1 = None
        copy: "i64[4, 511]" = torch.ops.aten.copy.default(slice_2, clone);  slice_2 = clone = None
        slice_scatter: "i64[4, 512]" = torch.ops.aten.slice_scatter.default(full, copy, 1, 1, 9223372036854775807);  full = copy = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:64 in shift_tokens_right, code: shifted_input_ids[:, 0] = decoder_start_token_id
        select_1: "i64[4]" = torch.ops.aten.select.int(slice_scatter, 1, 0)
        full_default: "i64[]" = torch.ops.aten.full.default([], 2, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_1: "i64[4]" = torch.ops.aten.copy.default(select_1, full_default);  select_1 = full_default = None
        select_scatter: "i64[4, 512]" = torch.ops.aten.select_scatter.default(slice_scatter, copy_1, 1, 0);  slice_scatter = copy_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:69 in shift_tokens_right, code: shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        eq: "b8[4, 512]" = torch.ops.aten.eq.Scalar(select_scatter, -100)
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4, 512]" = torch.ops.aten.where.self(eq, full_default_1, select_scatter);  eq = full_default_1 = select_scatter = None
        return (where,)
