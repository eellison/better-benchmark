"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train
Pattern hash: 5e8fb91fe0df
Shape hash: 32fa6fe1
"""
_shapes_config = "(T([8008, 2560], f32), T([32, 128], i64))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_1: "f32[8008, 2560]", primals_2: "i64[32, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:96 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[32, 128, 2560]" = torch.ops.aten.embedding.default(primals_1, primals_2, 0);  primals_1 = primals_2 = None
        mul_tensor: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([8008, 2560], dtype=torch.float32, device='cuda'),
    torch.randint(0, 8008, [32, 128], dtype=torch.int64, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
