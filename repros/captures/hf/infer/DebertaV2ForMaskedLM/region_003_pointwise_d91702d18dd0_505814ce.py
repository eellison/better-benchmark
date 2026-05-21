"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer
Pattern hash: d91702d18dd0
Shape hash: 505814ce
"""
_shapes_config = "(T([4096, 6144], f32), T([1536, 6144], f32), S([8, 512, 6144]), S([4096, 6144]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_142: "f32[4096, 6144]", arg386_1: "f32[1536, 6144]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_142, _shape_param_0);  addmm_142 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f32[6144, 1536]" = torch.ops.aten.permute.default(arg386_1, [1, 0]);  arg386_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([4096, 6144], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 6144], dtype=torch.float32, device='cuda'),
    [8, 512, 6144],  # _shape_param_0
    [4096, 6144],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
