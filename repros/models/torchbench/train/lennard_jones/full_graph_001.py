class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[1000, 1]", primals_4: "f32[16, 16]", primals_6: "f32[16, 16]", primals_8: "f32[16, 16]", primals_10: "f32[1, 16]", tanh: "f32[1000, 16]", tanh_1: "f32[1000, 16]", tanh_2: "f32[1000, 16]", tanh_3: "f32[1000, 16]", tangents_1: "f32[1000, 1]"):
        # No stacktrace found for following nodes
        permute_4: "f32[16, 1]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_5: "f32[1, 16]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mul_3: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(tangents_1, permute_5);  permute_5 = None
        permute_6: "f32[1, 1000]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm: "f32[1, 16]" = torch.ops.aten.mm.default(permute_6, tanh_3);  permute_6 = None
        sum_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view: "f32[1]" = torch.ops.aten.reshape.default(sum_1, [1]);  sum_1 = None
        mul_4: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub: "f32[1000, 16]" = torch.ops.aten.sub.Tensor(1, mul_4);  mul_4 = None
        mul_5: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mul_3, sub);  mul_3 = sub = None
        permute_3: "f32[16, 16]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_9: "f32[16, 16]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_1: "f32[1000, 16]" = torch.ops.aten.mm.default(mul_5, permute_9);  permute_9 = None
        permute_10: "f32[16, 1000]" = torch.ops.aten.permute.default(mul_5, [1, 0])
        mm_2: "f32[16, 16]" = torch.ops.aten.mm.default(permute_10, tanh_2);  permute_10 = None
        sum_2: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(mul_5, [0], True);  mul_5 = None
        view_1: "f32[16]" = torch.ops.aten.reshape.default(sum_2, [16]);  sum_2 = None
        mul_6: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_1: "f32[1000, 16]" = torch.ops.aten.sub.Tensor(1, mul_6);  mul_6 = None
        mul_7: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mm_1, sub_1);  mm_1 = sub_1 = None
        permute_2: "f32[16, 16]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_13: "f32[16, 16]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_3: "f32[1000, 16]" = torch.ops.aten.mm.default(mul_7, permute_13);  permute_13 = None
        permute_14: "f32[16, 1000]" = torch.ops.aten.permute.default(mul_7, [1, 0])
        mm_4: "f32[16, 16]" = torch.ops.aten.mm.default(permute_14, tanh_1);  permute_14 = None
        sum_3: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(mul_7, [0], True);  mul_7 = None
        view_2: "f32[16]" = torch.ops.aten.reshape.default(sum_3, [16]);  sum_3 = None
        mul_8: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_2: "f32[1000, 16]" = torch.ops.aten.sub.Tensor(1, mul_8);  mul_8 = None
        mul_9: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mm_3, sub_2);  mm_3 = sub_2 = None
        permute_1: "f32[16, 16]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_17: "f32[16, 16]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_5: "f32[1000, 16]" = torch.ops.aten.mm.default(mul_9, permute_17);  permute_17 = None
        permute_18: "f32[16, 1000]" = torch.ops.aten.permute.default(mul_9, [1, 0])
        mm_6: "f32[16, 16]" = torch.ops.aten.mm.default(permute_18, tanh);  permute_18 = None
        sum_4: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(mul_9, [0], True);  mul_9 = None
        view_3: "f32[16]" = torch.ops.aten.reshape.default(sum_4, [16]);  sum_4 = None
        mul_10: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_3: "f32[1000, 16]" = torch.ops.aten.sub.Tensor(1, mul_10);  mul_10 = None
        mul_11: "f32[1000, 16]" = torch.ops.aten.mul.Tensor(mm_5, sub_3);  mm_5 = sub_3 = None
        permute_21: "f32[16, 1000]" = torch.ops.aten.permute.default(mul_11, [1, 0])
        mm_7: "f32[16, 1]" = torch.ops.aten.mm.default(permute_21, primals_3);  permute_21 = primals_3 = None
        sum_5: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(mul_11, [0], True);  mul_11 = None
        view_4: "f32[16]" = torch.ops.aten.reshape.default(sum_5, [16]);  sum_5 = None
        return (mm_7, view_4, None, mm_6, view_3, mm_4, view_2, mm_2, view_1, mm, view)
