class GraphModule(torch.nn.Module):
    def forward(self, mm_22: "f32[6272, 2048]", _shape_param_0, primals_331: "f32[2048]", mul_224: "f32[128, 7, 7, 2048]", div_76: "f32[128, 7, 7, 1]", _shape_param_1, _shape_param_2, _shape_param_3, lt_41: "b8[128, 1, 1]", _shape_param_4, primals_329: "f32[512, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        reshape_default: "f32[128, 7, 7, 2048]" = torch.ops.aten.reshape.default(mm_22, _shape_param_0);  mm_22 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_tensor: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(reshape_default, primals_331);  reshape_default = primals_331 = None
        mul_tensor_1: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 2048)
        sum_dim_int_list: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_224);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_224, sum_dim_int_list_1);  mul_224 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(div_76, sub_tensor_1);  div_76 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        reshape_default_1: "f32[128, 7, 7, 2, 2, 512]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        permute_default: "f32[128, 7, 2, 7, 2, 512]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 4, 2, 3, 5]);  reshape_default_1 = None
        clone_default: "f32[128, 7, 2, 7, 2, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        reshape_default_3: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_41, torch.float32);  lt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9086956530809402);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_tensor_5: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(reshape_default_3, div_tensor);  reshape_default_3 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_4: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_4);  mul_tensor_5 = _shape_param_4 = None
        permute_default_1: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
        permute_default_2: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_4, permute_default_2)
