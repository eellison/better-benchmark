"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: e14c3479f4c6
Shape hash: 39e9e91d
"""
_shapes_config = "(T([16384, 49, 49], f32), T([49, 49], i64), T([169, 8], f32), T([2048, 8, 49, 32], f32, stride=(37632, 32, 768, 1)), S([2048, 8, 49, 49]), S([49, 49, -1]), S([2048, 8, 49, 49]), S([16384, 49, 49]), S([2048, 8, 49, 32]), S([16384, 49, 32]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_4: "f32[16384, 49, 49]", arg42_1: "i64[49, 49]", arg41_1: "f32[169, 8]", getitem_22: "f32[2048, 8, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default: "f32[2048, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_4, _shape_param_0);  bmm_4 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_1: "i64[2401]" = torch.ops.aten.reshape.default(arg42_1, [-1]);  arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_tensor: "f32[2401, 8]" = torch.ops.aten.index.Tensor(arg41_1, [reshape_default_1]);  arg41_1 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_2: "f32[49, 49, 8]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_1);  index_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default: "f32[8, 49, 49]" = torch.ops.aten.permute.default(reshape_default_2, [2, 0, 1]);  reshape_default_2 = None
        clone_default: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_default: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_tensor: "f32[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default, unsqueeze_default);  reshape_default = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_default: "f32[2048, 8, 49, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_default: "f32[2048, 8, 49, 49]" = torch.ops.aten.expand.default(div_tensor, _shape_param_2);  div_tensor = _shape_param_2 = None
        reshape_default_3: "f32[16384, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None
        expand_default_1: "f32[2048, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_22, _shape_param_4);  getitem_22 = _shape_param_4 = None
        clone_default_1: "f32[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[16384, 49, 32]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        return (reshape_default_3, reshape_default_4)



def make_inputs():
    return [
    torch.randn([16384, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randint(0, 169, [49, 49], dtype=torch.int64, device='cuda'),
    torch.randn([169, 8], dtype=torch.float32, device='cuda'),
    torch.randn(77069824, dtype=torch.float32, device='cuda').as_strided([2048, 8, 49, 32], [37632, 32, 768, 1]),  # getitem_22
    [2048, 8, 49, 49],  # _shape_param_0
    [49, 49, -1],  # _shape_param_1
    [2048, 8, 49, 49],  # _shape_param_2
    [16384, 49, 49],  # _shape_param_3
    [2048, 8, 49, 32],  # _shape_param_4
    [16384, 49, 32],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
