"""
Standalone repro captured via capture_hook.
Label: hf_PLBartForCausalLM_train
Pattern hash: 532b61911f41
Shape hash: c2952a44
"""
_shapes_config = "(T([50005, 768], f32), T([16, 1024, 768], f32), S([16384, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[50005, 768]", primals_1: "f32[16, 1024, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1125 in torch_dynamo_resume_in_forward_at_1111, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "f32[768, 50005]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(primals_1, _shape_param_0);  primals_1 = _shape_param_0 = None
        constant_pad_nd_default: "f32[768, 50008]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
        return (reshape_default, constant_pad_nd_default)



def make_inputs():
    return [
    torch.randn([50005, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1024, 768], dtype=torch.float32, device='cuda'),
    [16384, 768],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
