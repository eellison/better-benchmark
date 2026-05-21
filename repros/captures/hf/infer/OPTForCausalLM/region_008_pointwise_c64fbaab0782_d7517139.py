"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_infer
Pattern hash: c64fbaab0782
Shape hash: d7517139
"""
_shapes_config = "()"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:347 in forward, code: attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)
        full_default: "f32[4, 2048]" = torch.ops.aten.full.default([4, 2048], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return full_default



def make_inputs():
    return [

    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
