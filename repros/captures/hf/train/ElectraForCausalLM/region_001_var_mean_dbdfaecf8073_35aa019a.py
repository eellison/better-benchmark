"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_train
Pattern hash: dbdfaecf8073
Shape hash: 35aa019a
"""
_shapes_config = "(T([32768, 128], f32), T([128], f32), T([128], f32), T([30522, 128], f32), T([30522], f32), S([64, 512, 128]), S([32768, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_73: "f32[32768, 128]", primals_206: "f32[128]", primals_207: "f32[128]", primals_5: "f32[30522, 128]", primals_208: "f32[30522]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        reshape_default: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, _shape_param_0);  addmm_73 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[64, 512, 128]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(mul_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_2, getitem_1);  mul_tensor_2 = getitem_1 = None
        mul_tensor_3: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_4: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_3, primals_206);  mul_tensor_3 = primals_206 = None
        add_tensor_2: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_4, primals_207);  mul_tensor_4 = primals_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        reshape_default_1: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[128, 30522]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        constant_pad_nd_default: "f32[128, 30524]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 2, 0, 0]);  permute_default = None
        full_default: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[30524]" = torch.ops.aten.cat.default([primals_208, full_default]);  primals_208 = full_default = None
        return (reshape_default_1, constant_pad_nd_default, cat_default)



def make_inputs():
    return [
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([30522, 128], dtype=torch.float32, device='cuda'),
    torch.randn([30522], dtype=torch.float32, device='cuda'),
    [64, 512, 128],  # _shape_param_0
    [32768, 128],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
