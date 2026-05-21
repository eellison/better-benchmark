"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 7d4fb3918223
Shape hash: 3cd993ce
"""
_shapes_config = "(T([8192, 3072], f32), T([8192, 3072], f32), T([3072, 768], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu: "f32[8192, 3072]", mm: "f32[8192, 3072]", primals_15: "f32[3072, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        le_scalar: "b8[8192, 3072]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        where_self: "f32[8192, 3072]" = torch.ops.aten.where.self(le_scalar, full_default, mm);  le_scalar = full_default = mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_default_1: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (where_self, permute_default_1)



def make_inputs():
    return [
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
