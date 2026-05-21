class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1152000, 512]", primals_2: "f32[512]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:384 in layernorm_bwd, code: x_f32 = x.float()
        convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32)

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:385 in layernorm_bwd, code: out = F.layer_norm(x_f32, w.shape, w, None, 1e-6).to(x.dtype)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [1], correction = 0, keepdim = True)
        getitem: "f32[1152000, 1]" = var_mean[0]
        getitem_1: "f32[1152000, 1]" = var_mean[1];  var_mean = None
        add: "f32[1152000, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[1152000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, primals_2);  mul = None
        convert_element_type_1: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:386 in layernorm_bwd, code: return out.sum()
        sum_1: "bf16[]" = torch.ops.aten.sum.default(convert_element_type_1);  convert_element_type_1 = None
        return (sum_1, primals_1, primals_2, getitem_1, rsqrt)
