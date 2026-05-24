import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 128]", arg1_1: "f32[128, 32000]", arg2_1: "i64[1]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:72 in torch_dynamo_resume_in__dequeue_and_enqueue_at_68, code: self.queue[:, ptr : ptr + batch_size] = keys.T
        slice_1: "f32[128, 32]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 32)
        permute: "f32[128, 32]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        copy: "f32[128, 32]" = torch.ops.aten.copy.default(slice_1, permute);  slice_1 = permute = None
        slice_scatter: "f32[128, 32000]" = torch.ops.aten.slice_scatter.default(arg1_1, copy, 1, 0, 32);  copy = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:75 in torch_dynamo_resume_in__dequeue_and_enqueue_at_68, code: self.queue_ptr[0] = ptr
        _tensor_constant0: "i64[]" = self._tensor_constant0;  _tensor_constant0 = None
        select: "i64[]" = torch.ops.aten.select.int(arg2_1, 0, 0)
        full_default: "i64[]" = torch.ops.aten.full.default([], 32, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_1: "i64[]" = torch.ops.aten.copy.default(select, full_default);  select = full_default = None
        select_scatter: "i64[1]" = torch.ops.aten.select_scatter.default(arg2_1, copy_1, 0, 0);  copy_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:72 in torch_dynamo_resume_in__dequeue_and_enqueue_at_68, code: self.queue[:, ptr : ptr + batch_size] = keys.T
        copy_: "f32[128, 32000]" = torch.ops.aten.copy_.default(arg1_1, slice_scatter);  arg1_1 = slice_scatter = copy_ = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:75 in torch_dynamo_resume_in__dequeue_and_enqueue_at_68, code: self.queue_ptr[0] = ptr
        copy__1: "i64[1]" = torch.ops.aten.copy_.default(arg2_1, select_scatter);  arg2_1 = select_scatter = copy__1 = None
        return ()
