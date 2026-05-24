import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:826 in document_causal, code: causal_mask = q_idx >= kv_idx
        ge: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:827 in document_causal, code: document_mask = docs[q_idx] == docs[kv_idx]
        index: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
        index_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
        eq: "b8[]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:828 in document_causal, code: return causal_mask & document_mask
        bitwise_and: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge, eq);  ge = eq = None
        return bitwise_and
