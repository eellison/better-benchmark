"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train
Pattern hash: edf2e0dc44ea
Shape hash: d999c84a
"""
_shapes_config = "(T([128, 2560], f32), T([128], i64))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[128, 2560]", primals_1: "i64[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_default: "f32[128, 2560]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = primals_1 = None
        return embedding_default



def make_inputs():
    return [
    torch.randn([128, 2560], dtype=torch.float32, device='cuda'),
    torch.randint(0, 128, [128], dtype=torch.int64, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
