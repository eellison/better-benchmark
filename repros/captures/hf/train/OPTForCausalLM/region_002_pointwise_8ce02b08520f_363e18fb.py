"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 8ce02b08520f
Shape hash: 363e18fb
"""
_shapes_config = "(T([2], i64), T([8192, 768], f32), T([8192, 768], f32), S([4, 2048, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, inductor_seeds_default: "i64[2]", addmm_5: "f32[8192, 768]", view_12: "f32[8192, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:245 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[8192, 768]" = torch.ops.prims.inductor_random.default([8192, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8192, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, addmm_5);  gt_scalar = addmm_5 = None
        mul_tensor_1: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_tensor: "f32[8192, 768]" = torch.ops.aten.add.Tensor(view_12, mul_tensor_1);  view_12 = mul_tensor_1 = None
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_0);  add_tensor = _shape_param_0 = None
        return reshape_default



def make_inputs():
    return [
    torch.randint(0, 2, [2], dtype=torch.int64, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [4, 2048, 768],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
