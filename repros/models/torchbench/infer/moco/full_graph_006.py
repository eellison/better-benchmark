import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32]", arg1_1: "f32[32, 3, 224, 224]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:97 in torch_dynamo_resume_in__batch_shuffle_ddp_at_94, code: idx_unshuffle = torch.argsort(idx_shuffle)
        sort = torch.ops.aten.sort.default(arg0_1)
        getitem_1: "i64[32]" = sort[1];  sort = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:101 in torch_dynamo_resume_in__batch_shuffle_ddp_at_94, code: idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        view: "i64[1, 32]" = torch.ops.aten.reshape.default(arg0_1, [1, -1]);  arg0_1 = None
        select: "i64[32]" = torch.ops.aten.select.int(view, 0, 0);  view = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:103 in torch_dynamo_resume_in__batch_shuffle_ddp_at_94, code: return x_gather[idx_this], idx_unshuffle
        index: "f32[32, 3, 224, 224]" = torch.ops.aten.index.Tensor(arg1_1, [select]);  arg1_1 = select = None
        return (index, getitem_1)
