"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_infer
Pattern hash: 84b26f7cd441
Shape hash: a92ccc01
"""
_shapes_config = "(T([50400, 4096], f32), T([1, 128], i64), T([4096], f32), T([4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), S([128, 4096]), S([128, 4096]), S([128, 4096]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50400, 4096]", arg0_1: "i64[1, 128]", arg2_1: "f32[4096]", arg3_1: "f32[4096]", arg4_1: "f32[4096, 4096]", arg5_1: "f32[4096, 4096]", arg6_1: "f32[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:494 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding_default: "f32[1, 128, 4096]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(embedding_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 128, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:502 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:503 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_tensor: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding_default, getitem_1);  embedding_default = getitem_1 = None
        add_tensor_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, arg2_1);  mul_tensor = arg2_1 = None
        add_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg3_1);  mul_tensor_1 = arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        reshape_default: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        reshape_default_1: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        reshape_default_2: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        permute_default_2: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_tensor: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze_default, 1, 0, 1)
        sub_tensor_1: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat_default: "i64[1, 129]" = torch.ops.aten.cat.default([sub_tensor_1, unsqueeze_default], -1);  sub_tensor_1 = unsqueeze_default = None
        slice_tensor_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 129)
        slice_tensor_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 128);  cat_default = None
        sub_tensor_2: "i64[1, 128]" = torch.ops.aten.sub.Tensor(slice_tensor_1, slice_tensor_2);  slice_tensor_1 = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne_scalar: "b8[1, 128]" = torch.ops.aten.ne.Scalar(sub_tensor_2, 1);  sub_tensor_2 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2, ne_scalar)



def make_inputs():
    return [
    torch.randn([50400, 4096], dtype=torch.float32, device='cuda'),
    torch.randint(0, 50400, [1, 128], dtype=torch.int64, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    [128, 4096],  # _shape_param_0
    [128, 4096],  # _shape_param_1
    [128, 4096],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
