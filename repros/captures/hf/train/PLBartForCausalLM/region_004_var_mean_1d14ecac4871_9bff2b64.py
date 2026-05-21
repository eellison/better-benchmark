"""
Standalone repro captured via capture_hook.
Label: hf_PLBartForCausalLM_train
Pattern hash: 1d14ecac4871
Shape hash: 9bff2b64
"""
_shapes_config = "(T([16384, 768], f32), T([2], i64), T([16, 1024, 768], f32), T([768], f32), T([768], f32), S([16, 1024, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[16384, 768]", inductor_seeds_default: "i64[2]", add_2: "f32[16, 1024, 768]", primals_17: "f32[768]", primals_18: "f32[768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:476 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[16, 1024, 768]" = torch.ops.aten.reshape.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:477 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 1024, 768]" = torch.ops.prims.inductor_random.default([16, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[16, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:478 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[16, 1024, 768]" = torch.ops.aten.add.Tensor(add_2, mul_tensor_1);  add_2 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:479 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[16, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[16, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[16, 1024, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_17);  mul_tensor_2 = primals_17 = None
        add_tensor_2: "f32[16, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_18);  mul_tensor_3 = primals_18 = None
        div_tensor: "f32[16, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        return (add_tensor_2, div_tensor)



def make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [2], dtype=torch.int64, device='cuda'),
    torch.randn([16, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [16, 1024, 768],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
