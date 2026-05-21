"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_train
Pattern hash: 8297ca6e8fa9
Shape hash: 73991007
"""
_shapes_config = "(T([32, 12, 512, 64], f32, stride=(393216, 64, 768, 1)), T([768, 768], f32), S([32, 512, -1]), S([16384, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_52: "f32[32, 12, 512, 64]", primals_194: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
        clone_default: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn(12582912, dtype=torch.float32, device='cuda').as_strided([32, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_52
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [32, 512, -1],  # _shape_param_0
    [16384, 768],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
