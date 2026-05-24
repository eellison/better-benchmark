import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[2048, 512]", primals_4: "f32[64, 512]", primals_8: "i64[204790]", primals_10: "i64[204789]", primals_12: "i64[204793]", primals_14: "i64[204790]", primals_16: "i64[204793]", primals_18: "i64[204784]", primals_20: "i64[204786]", primals_22: "i64[204792]", primals_23: "f32[1024, 100]", primals_25: "f32[1024, 1024]", primals_27: "f32[1024, 1024]", primals_29: "f32[1, 1024]", relu: "f32[2048, 512]", select: "i64[2048]", getitem_1: "i64[204790]", getitem_2: "i64[2048]", getitem_3: "i64[0]", select_1: "i64[2048]", getitem_5: "i64[204789]", getitem_6: "i64[2048]", getitem_7: "i64[0]", select_2: "i64[2048]", getitem_9: "i64[204793]", getitem_10: "i64[2048]", getitem_11: "i64[0]", select_3: "i64[2048]", getitem_13: "i64[204790]", getitem_14: "i64[2048]", getitem_15: "i64[0]", select_4: "i64[2048]", getitem_17: "i64[204793]", getitem_18: "i64[2048]", getitem_19: "i64[0]", select_5: "i64[2048]", getitem_21: "i64[204784]", getitem_22: "i64[2048]", getitem_23: "i64[0]", select_6: "i64[2048]", getitem_25: "i64[204786]", getitem_26: "i64[2048]", getitem_27: "i64[0]", select_7: "i64[2048]", getitem_29: "i64[204792]", getitem_30: "i64[2048]", getitem_31: "i64[0]", permute_2: "f32[2048, 64, 9]", cat_1: "f32[2048, 100]", relu_2: "f32[2048, 1024]", relu_3: "f32[2048, 1024]", relu_4: "f32[2048, 1024]", le: "b8[2048, 1]", le_4: "b8[2048, 64]", tangents_1: "f32[2048, 1]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[2048, 1]" = torch.ops.aten.where.self(le, full_default, tangents_1);  le = tangents_1 = None
        permute_6: "f32[1024, 1]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_7: "f32[1, 1024]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        mul: "f32[2048, 1024]" = torch.ops.aten.mul.Tensor(where, permute_7);  permute_7 = None
        permute_8: "f32[1, 2048]" = torch.ops.aten.permute.default(where, [1, 0])
        mm: "f32[1, 1024]" = torch.ops.aten.mm.default(permute_8, relu_4);  permute_8 = None
        sum_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        view_1: "f32[1]" = torch.ops.aten.reshape.default(sum_1, [1]);  sum_1 = None
        le_1: "b8[2048, 1024]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_1: "f32[2048, 1024]" = torch.ops.aten.where.self(le_1, full_default, mul);  le_1 = mul = None
        permute_5: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_1: "f32[2048, 1024]" = torch.ops.aten.mm.default(where_1, permute_11);  permute_11 = None
        permute_12: "f32[1024, 2048]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_2: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_12, relu_3);  permute_12 = None
        sum_2: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        view_2: "f32[1024]" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
        le_2: "b8[2048, 1024]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_2: "f32[2048, 1024]" = torch.ops.aten.where.self(le_2, full_default, mm_1);  le_2 = mm_1 = None
        permute_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_15: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_3: "f32[2048, 1024]" = torch.ops.aten.mm.default(where_2, permute_15);  permute_15 = None
        permute_16: "f32[1024, 2048]" = torch.ops.aten.permute.default(where_2, [1, 0])
        mm_4: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_16, relu_2);  permute_16 = None
        sum_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_2, [0], True);  where_2 = None
        view_3: "f32[1024]" = torch.ops.aten.reshape.default(sum_3, [1024]);  sum_3 = None
        le_3: "b8[2048, 1024]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_3: "f32[2048, 1024]" = torch.ops.aten.where.self(le_3, full_default, mm_3);  le_3 = mm_3 = None
        permute_3: "f32[100, 1024]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_19: "f32[1024, 100]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_5: "f32[2048, 100]" = torch.ops.aten.mm.default(where_3, permute_19);  permute_19 = None
        permute_20: "f32[1024, 2048]" = torch.ops.aten.permute.default(where_3, [1, 0])
        mm_6: "f32[1024, 100]" = torch.ops.aten.mm.default(permute_20, cat_1);  permute_20 = cat_1 = None
        sum_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(where_3, [0], True);  where_3 = None
        view_4: "f32[1024]" = torch.ops.aten.reshape.default(sum_4, [1024]);  sum_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:337 in interact_features, code: R = torch.cat([x] + [Zflat], dim=1)
        slice_1: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(mm_5, 1, 0, 64)
        slice_2: "f32[2048, 36]" = torch.ops.aten.slice.Tensor(mm_5, 1, 64, 100);  mm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:335 in interact_features, code: Zflat = Z[:, li, lj]
        full_default_4: "f32[2048, 9, 9]" = torch.ops.aten.full.default([2048, 9, 9], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:329 in interact_features, code: li = torch.tensor(
        _tensor_constant0: "i64[36]" = self._tensor_constant0
        lift_fresh_copy: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:332 in interact_features, code: lj = torch.tensor(
        _tensor_constant1: "i64[36]" = self._tensor_constant1
        lift_fresh_copy_1: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:335 in interact_features, code: Zflat = Z[:, li, lj]
        index_put: "f32[2048, 9, 9]" = torch.ops.aten.index_put.default(full_default_4, [None, lift_fresh_copy, lift_fresh_copy_1], slice_2, True);  full_default_4 = lift_fresh_copy = lift_fresh_copy_1 = slice_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:318 in interact_features, code: Z = torch.bmm(T, torch.transpose(T, 1, 2))
        bmm_1: "f32[2048, 64, 9]" = torch.ops.aten.bmm.default(permute_2, index_put)
        permute_24: "f32[2048, 9, 64]" = torch.ops.aten.permute.default(permute_2, [0, 2, 1]);  permute_2 = None
        bmm_2: "f32[2048, 9, 64]" = torch.ops.aten.bmm.default(index_put, permute_24);  index_put = permute_24 = None
        permute_25: "f32[2048, 9, 64]" = torch.ops.aten.permute.default(bmm_1, [0, 2, 1]);  bmm_1 = None
        add: "f32[2048, 9, 64]" = torch.ops.aten.add.Tensor(bmm_2, permute_25);  bmm_2 = permute_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:316 in interact_features, code: T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        view_5: "f32[2048, 576]" = torch.ops.aten.reshape.default(add, [2048, 576]);  add = None
        slice_3: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_5, 1, 0, 64)
        slice_4: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_5, 1, 64, 128)
        slice_5: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_5, 1, 128, 192)
        slice_6: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_5, 1, 192, 256)
        slice_7: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_5, 1, 256, 320)
        slice_8: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_5, 1, 320, 384)
        slice_9: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_5, 1, 384, 448)
        slice_10: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_5, 1, 448, 512)
        slice_11: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(view_5, 1, 512, 576);  view_5 = None
        add_1: "f32[2048, 64]" = torch.ops.aten.add.Tensor(slice_1, slice_3);  slice_1 = slice_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_backward: "f32[1000000, 64]" = torch.ops.aten._embedding_bag_backward.default(slice_11, primals_22, select_7, getitem_29, getitem_30, getitem_31, 1000000, False, 0, True, None);  slice_11 = primals_22 = select_7 = getitem_29 = getitem_30 = getitem_31 = None
        _embedding_bag_backward_1: "f32[1000000, 64]" = torch.ops.aten._embedding_bag_backward.default(slice_10, primals_20, select_6, getitem_25, getitem_26, getitem_27, 1000000, False, 0, True, None);  slice_10 = primals_20 = select_6 = getitem_25 = getitem_26 = getitem_27 = None
        _embedding_bag_backward_2: "f32[1000000, 64]" = torch.ops.aten._embedding_bag_backward.default(slice_9, primals_18, select_5, getitem_21, getitem_22, getitem_23, 1000000, False, 0, True, None);  slice_9 = primals_18 = select_5 = getitem_21 = getitem_22 = getitem_23 = None
        _embedding_bag_backward_3: "f32[1000000, 64]" = torch.ops.aten._embedding_bag_backward.default(slice_8, primals_16, select_4, getitem_17, getitem_18, getitem_19, 1000000, False, 0, True, None);  slice_8 = primals_16 = select_4 = getitem_17 = getitem_18 = getitem_19 = None
        _embedding_bag_backward_4: "f32[1000000, 64]" = torch.ops.aten._embedding_bag_backward.default(slice_7, primals_14, select_3, getitem_13, getitem_14, getitem_15, 1000000, False, 0, True, None);  slice_7 = primals_14 = select_3 = getitem_13 = getitem_14 = getitem_15 = None
        _embedding_bag_backward_5: "f32[1000000, 64]" = torch.ops.aten._embedding_bag_backward.default(slice_6, primals_12, select_2, getitem_9, getitem_10, getitem_11, 1000000, False, 0, True, None);  slice_6 = primals_12 = select_2 = getitem_9 = getitem_10 = getitem_11 = None
        _embedding_bag_backward_6: "f32[1000000, 64]" = torch.ops.aten._embedding_bag_backward.default(slice_5, primals_10, select_1, getitem_5, getitem_6, getitem_7, 1000000, False, 0, True, None);  slice_5 = primals_10 = select_1 = getitem_5 = getitem_6 = getitem_7 = None
        _embedding_bag_backward_7: "f32[1000000, 64]" = torch.ops.aten._embedding_bag_backward.default(slice_4, primals_8, select, getitem_1, getitem_2, getitem_3, 1000000, False, 0, True, None);  slice_4 = primals_8 = select = getitem_1 = getitem_2 = getitem_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        where_4: "f32[2048, 64]" = torch.ops.aten.where.self(le_4, full_default, add_1);  le_4 = add_1 = None
        permute_1: "f32[512, 64]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_26: "f32[64, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_7: "f32[2048, 512]" = torch.ops.aten.mm.default(where_4, permute_26);  permute_26 = None
        permute_27: "f32[64, 2048]" = torch.ops.aten.permute.default(where_4, [1, 0])
        mm_8: "f32[64, 512]" = torch.ops.aten.mm.default(permute_27, relu);  permute_27 = None
        sum_5: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(where_4, [0], True);  where_4 = None
        view_6: "f32[64]" = torch.ops.aten.reshape.default(sum_5, [64]);  sum_5 = None
        le_5: "b8[2048, 512]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_5: "f32[2048, 512]" = torch.ops.aten.where.self(le_5, full_default, mm_7);  le_5 = full_default = mm_7 = None
        permute_30: "f32[512, 2048]" = torch.ops.aten.permute.default(where_5, [1, 0])
        mm_9: "f32[512, 512]" = torch.ops.aten.mm.default(permute_30, primals_3);  permute_30 = primals_3 = None
        sum_6: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(where_5, [0], True);  where_5 = None
        view_7: "f32[512]" = torch.ops.aten.reshape.default(sum_6, [512]);  sum_6 = None
        return (mm_9, view_7, None, mm_8, view_6, None, _embedding_bag_backward_7, None, _embedding_bag_backward_6, None, _embedding_bag_backward_5, None, _embedding_bag_backward_4, None, _embedding_bag_backward_3, None, _embedding_bag_backward_2, None, _embedding_bag_backward_1, None, _embedding_bag_backward, None, mm_6, view_4, mm_4, view_3, mm_2, view_2, mm, view_1)
