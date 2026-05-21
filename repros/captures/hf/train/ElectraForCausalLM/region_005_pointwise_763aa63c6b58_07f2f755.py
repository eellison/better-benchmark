"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_train
Pattern hash: 763aa63c6b58
Shape hash: 07f2f755
"""
_shapes_config = "(T([256, 512, 64], f32), T([256, 256], f32), S([64, 4, 512, 64]), S([64, 512, -1]), S([32768, 256]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_23: "f32[256, 512, 64]", primals_194: "f32[256, 256]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, _shape_param_0);  bmm_23 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_1: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_2: "f32[32768, 256]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[256, 256]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([256, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([256, 256], dtype=torch.float32, device='cuda'),
    [64, 4, 512, 64],  # _shape_param_0
    [64, 512, -1],  # _shape_param_1
    [32768, 256],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
