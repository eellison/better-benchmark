import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 64, 16, 16]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in torch_dynamo_resume_in__forward_impl_at_269, code: x = self.relu(x)
        relu: "f32[64, 64, 16, 16]" = torch.ops.aten.relu.default(arg0_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in torch_dynamo_resume_in__forward_impl_at_269, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False)
        getitem: "f32[64, 64, 8, 8]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in torch_dynamo_resume_in__forward_impl_at_269, code: x = self.relu(x)
        copy_: "f32[64, 64, 16, 16]" = torch.ops.aten.copy_.default(arg0_1, relu);  arg0_1 = relu = copy_ = None
        return (getitem,)
