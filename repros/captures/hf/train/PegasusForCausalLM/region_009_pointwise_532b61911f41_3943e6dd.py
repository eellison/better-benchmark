"""
Standalone repro captured via capture_hook.
Label: hf_PegasusForCausalLM_train
Pattern hash: 532b61911f41
Shape hash: 3943e6dd
"""
_shapes_config = "(T([50265, 1024], f32), T([128, 128, 1024], f32), S([16384, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[50265, 1024]", primals_1: "f32[128, 128, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:1114 in torch_dynamo_resume_in_forward_at_1100, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "f32[1024, 50265]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        reshape_default: "f32[16384, 1024]" = torch.ops.aten.reshape.default(primals_1, _shape_param_0);  primals_1 = _shape_param_0 = None
        constant_pad_nd_default: "f32[1024, 50268]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
        return (reshape_default, constant_pad_nd_default)



def make_inputs():
    return [
    torch.randn([50265, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 1024], dtype=torch.float32, device='cuda'),
    [16384, 1024],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
