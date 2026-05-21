"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 3c8a33a68547
Shape hash: ada934d6
"""
_shapes_config = "(T([512, 128, 80], f32), T([2560, 2560], f32), S([16, 32, 128, 80]), S([16, 128, 2560]), S([2048, 2560]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_5: "f32[512, 128, 80]", primals_16: "f32[2560, 2560]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_5, _shape_param_0);  bmm_5 = _shape_param_0 = None
        mul_scalar: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(reshape_default, 0.334370152488211);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(mul_scalar, [0, 2, 1, 3]);  mul_scalar = None
        clone_default: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[2048, 2560]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_default_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_2, permute_default_2)



def make_inputs():
    return [
    torch.randn([512, 128, 80], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [16, 32, 128, 80],  # _shape_param_0
    [16, 128, 2560],  # _shape_param_1
    [2048, 2560],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
