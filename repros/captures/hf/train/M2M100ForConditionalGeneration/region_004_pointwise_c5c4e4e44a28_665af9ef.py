"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train
Pattern hash: c5c4e4e44a28
Shape hash: 665af9ef
"""
_shapes_config = "(T([8192, 1024], f32), T([3], i64), T([64, 128, 1024], f32), T([64, 128, 4096], f32), T([1024, 128, 128], f32), T([1024, 128, 64], f32), T([1024, 128, 64], f32), T([1024, 64, 128], f32), S([64, 128, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[8192, 1024]", inductor_seeds_default: "i64[3]", add_3: "f32[64, 128, 1024]", relu: "f32[64, 128, 4096]", view_12: "f32[1024, 128, 128]", view_13: "f32[1024, 128, 64]", view_9: "f32[1024, 128, 64]", view_10: "f32[1024, 64, 128]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:376 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_random_default: "f32[64, 128, 1024]" = torch.ops.prims.inductor_random.default([64, 128, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[64, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:377 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_3, mul_tensor_1);  add_3 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        le_scalar: "b8[64, 128, 4096]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_default_1: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_default_2: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_3: "f32[1024, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (add_tensor, le_scalar, permute_default, permute_default_1, permute_default_2, permute_default_3)



def make_inputs():
    return [
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 3, [3], dtype=torch.int64, device='cuda'),
    torch.randn([64, 128, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 64, 128], dtype=torch.float32, device='cuda'),
    [64, 128, 1024],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
