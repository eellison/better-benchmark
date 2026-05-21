"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: a280b485c3f3
Shape hash: 225ecbc6
"""
_shapes_config = "(T([4096, 1536], f32), T([73], i64), T([8, 512, 1536], f32), T([1536], f32), T([1536], f32), T([1536, 1536], f32), S([8, 512, 1536]), S([4096, 1536]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_143: "f32[4096, 1536]", inductor_seeds_default: "i64[73]", add_166: "f32[8, 512, 1536]", primals_389: "f32[1536]", primals_390: "f32[1536]", primals_391: "f32[1536, 1536]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_143, _shape_param_0);  addmm_143 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 72);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_tensor: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_tensor_1, add_166);  mul_tensor_1 = add_166 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_389);  mul_tensor_2 = primals_389 = None
        add_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_390);  mul_tensor_3 = primals_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:818 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_391, [1, 0]);  primals_391 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([4096, 1536], dtype=torch.float32, device='cuda'),
    torch.randint(0, 73, [73], dtype=torch.int64, device='cuda'),
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
