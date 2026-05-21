"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train
Pattern hash: 1b4ae8212a8f
Shape hash: 367f2f63
"""
_shapes_config = "(T([4096, 2560], f32), T([2], i64), T([32, 128, 2560], f32), S([32, 128, 2560]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[4096, 2560]", inductor_seeds_default: "i64[2]", add_2: "f32[32, 128, 2560]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:391 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 128, 2560]" = torch.ops.prims.inductor_random.default([32, 128, 2560], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[32, 128, 2560]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(add_2, mul_tensor_1);  add_2 = mul_tensor_1 = None
        return add_tensor



def make_inputs():
    return [
    torch.randn([4096, 2560], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [2], dtype=torch.int64, device='cuda'),
    torch.randn([32, 128, 2560], dtype=torch.float32, device='cuda'),
    [32, 128, 2560],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
