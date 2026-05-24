import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[768, 768]", arg1_1: "f16[768]", arg2_1: "f16[1, 4096, 768]", arg3_1: "f16[768]", arg4_1: "f16[768]", arg5_1: "f16[50265, 768]", arg6_1: "f16[50265]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1276 in forward, code: x = self.dense(features)
        view: "f16[4096, 768]" = torch.ops.aten.reshape.default(arg2_1, [4096, 768]);  arg2_1 = None
        permute: "f16[768, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "f16[4096, 768]" = torch.ops.aten.addmm.default(arg1_1, view, permute);  arg1_1 = view = permute = None
        view_1: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(addmm, [1, 4096, 768]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_3: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.5)
        mul_1: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.7071067811865476);  convert_element_type_3 = None
        erf: "f32[1, 4096, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        convert_element_type_default: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float32);  mul_2 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 4096, 1]" = var_mean[0]
        getitem_1: "f32[1, 4096, 1]" = var_mean[1];  var_mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        full_default: "f16[7]" = torch.ops.aten.full.default([7], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f16[50272]" = torch.ops.aten.cat.default([arg6_1, full_default]);  arg6_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        sub: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_1: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_3: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_4: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
        add_2: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_4, arg4_1);  mul_4 = arg4_1 = None
        convert_element_type_6: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.float16);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        view_2: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_6, [4096, 768]);  convert_element_type_6 = None
        permute_1: "f16[768, 50265]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        constant_pad_nd_default: "f16[768, 50272]" = torch.ops.aten.constant_pad_nd.default(permute_1, [0, 7, 0, 0]);  permute_1 = None
        addmm_default: "f16[4096, 50272]" = torch.ops.aten.addmm.default(cat_default, view_2, constant_pad_nd_default);  cat_default = view_2 = constant_pad_nd_default = None
        slice_tensor: "f16[4096, 50265]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -7);  addmm_default = None
        view_3: "f16[1, 4096, 50265]" = torch.ops.aten.reshape.default(slice_tensor, [1, 4096, 50265]);  slice_tensor = None
        return (view_3,)
