"""
Standalone repro captured via capture_hook.
Label: hf_PegasusForCausalLM_train
Pattern hash: edf2e0dc44ea
Shape hash: 4494d7dc
"""
_shapes_config = "(T([1024, 1024], f32), T([128], i64))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[1024, 1024]", arg0_1: "i64[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:105 in forward, code: return super().forward(position_ids)
        embedding_default: "f32[128, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        return embedding_default



def make_inputs():
    return [
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 1024, [128], dtype=torch.int64, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
