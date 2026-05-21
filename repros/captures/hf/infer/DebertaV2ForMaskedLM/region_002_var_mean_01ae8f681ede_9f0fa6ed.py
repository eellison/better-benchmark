"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer
Pattern hash: 01ae8f681ede
Shape hash: 9f0fa6ed
"""
_shapes_config = "(T([4096, 1536], f32), T([8, 512, 1536], f32), T([1536], f32), T([1536], f32), T([1536, 1536], f32), S([8, 512, 1536]), S([4096, 1536]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_143: "f32[4096, 1536]", add_166: "f32[8, 512, 1536]", arg388_1: "f32[1536]", arg389_1: "f32[1536]", arg390_1: "f32[1536, 1536]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_143, _shape_param_0);  addmm_143 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_tensor: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(reshape_default, add_166);  reshape_default = add_166 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor, arg388_1);  mul_tensor = arg388_1 = None
        add_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg389_1);  mul_tensor_1 = arg389_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:818 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg390_1, [1, 0]);  arg390_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([4096, 1536], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1536], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 1536], dtype=torch.float32, device='cuda'),
    [8, 512, 1536],  # _shape_param_0
    [4096, 1536],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
