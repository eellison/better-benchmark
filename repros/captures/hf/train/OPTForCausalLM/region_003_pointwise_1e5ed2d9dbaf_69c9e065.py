"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 1e5ed2d9dbaf
Shape hash: 69c9e065
"""
_shapes_config = "(T([8192, 3072], f32), T([768, 3072], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_4: "f32[8192, 3072]", primals_17: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu_default: "f32[8192, 3072]" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        return (relu_default, permute_default)



def make_inputs():
    return [
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
