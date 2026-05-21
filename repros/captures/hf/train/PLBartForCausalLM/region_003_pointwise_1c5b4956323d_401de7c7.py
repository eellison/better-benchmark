"""
Standalone repro captured via capture_hook.
Label: hf_PLBartForCausalLM_train
Pattern hash: 1c5b4956323d
Shape hash: 401de7c7
"""
_shapes_config = "(T([1024], i64), T([1026, 768], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_1: "i64[1024]", primals_2: "f32[1026, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:112 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:114 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        embedding_default: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_2, add_tensor);  primals_2 = add_tensor = None
        return embedding_default



def make_inputs():
    return [
    torch.randint(0, 1024, [1024], dtype=torch.int64, device='cuda'),
    torch.randn([1026, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
