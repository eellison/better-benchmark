class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[512, 512]", primals_2: "f32[512]", primals_3: "f32[2048, 512]", primals_4: "f32[64, 512]", primals_5: "f32[64]", primals_6: "i64[8, 2048]", primals_7: "f32[1000000, 64]", primals_8: "i64[204790]", primals_9: "f32[1000000, 64]", primals_10: "i64[204789]", primals_11: "f32[1000000, 64]", primals_12: "i64[204793]", primals_13: "f32[1000000, 64]", primals_14: "i64[204790]", primals_15: "f32[1000000, 64]", primals_16: "i64[204793]", primals_17: "f32[1000000, 64]", primals_18: "i64[204784]", primals_19: "f32[1000000, 64]", primals_20: "i64[204786]", primals_21: "f32[1000000, 64]", primals_22: "i64[204792]", primals_23: "f32[1024, 100]", primals_24: "f32[1024]", primals_25: "f32[1024, 1024]", primals_26: "f32[1024]", primals_27: "f32[1024, 1024]", primals_28: "f32[1024]", primals_29: "f32[1, 1024]", primals_30: "f32[1]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        permute: "f32[512, 512]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm: "f32[2048, 512]" = torch.ops.aten.addmm.default(primals_2, primals_3, permute);  primals_2 = permute = None
        relu: "f32[2048, 512]" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1: "f32[512, 64]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "f32[2048, 64]" = torch.ops.aten.addmm.default(primals_5, relu, permute_1);  primals_5 = permute_1 = None
        relu_1: "f32[2048, 64]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select: "i64[2048]" = torch.ops.aten.select.int(primals_6, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag = torch.ops.aten._embedding_bag.default(primals_7, primals_8, select, False, 0, True);  primals_7 = None
        getitem: "f32[2048, 64]" = _embedding_bag[0]
        getitem_1: "i64[204790]" = _embedding_bag[1]
        getitem_2: "i64[2048]" = _embedding_bag[2]
        getitem_3: "i64[0]" = _embedding_bag[3];  _embedding_bag = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_1: "i64[2048]" = torch.ops.aten.select.int(primals_6, 0, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_1 = torch.ops.aten._embedding_bag.default(primals_9, primals_10, select_1, False, 0, True);  primals_9 = None
        getitem_4: "f32[2048, 64]" = _embedding_bag_1[0]
        getitem_5: "i64[204789]" = _embedding_bag_1[1]
        getitem_6: "i64[2048]" = _embedding_bag_1[2]
        getitem_7: "i64[0]" = _embedding_bag_1[3];  _embedding_bag_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_2: "i64[2048]" = torch.ops.aten.select.int(primals_6, 0, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_2 = torch.ops.aten._embedding_bag.default(primals_11, primals_12, select_2, False, 0, True);  primals_11 = None
        getitem_8: "f32[2048, 64]" = _embedding_bag_2[0]
        getitem_9: "i64[204793]" = _embedding_bag_2[1]
        getitem_10: "i64[2048]" = _embedding_bag_2[2]
        getitem_11: "i64[0]" = _embedding_bag_2[3];  _embedding_bag_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_3: "i64[2048]" = torch.ops.aten.select.int(primals_6, 0, 3)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_3 = torch.ops.aten._embedding_bag.default(primals_13, primals_14, select_3, False, 0, True);  primals_13 = None
        getitem_12: "f32[2048, 64]" = _embedding_bag_3[0]
        getitem_13: "i64[204790]" = _embedding_bag_3[1]
        getitem_14: "i64[2048]" = _embedding_bag_3[2]
        getitem_15: "i64[0]" = _embedding_bag_3[3];  _embedding_bag_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_4: "i64[2048]" = torch.ops.aten.select.int(primals_6, 0, 4)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_4 = torch.ops.aten._embedding_bag.default(primals_15, primals_16, select_4, False, 0, True);  primals_15 = None
        getitem_16: "f32[2048, 64]" = _embedding_bag_4[0]
        getitem_17: "i64[204793]" = _embedding_bag_4[1]
        getitem_18: "i64[2048]" = _embedding_bag_4[2]
        getitem_19: "i64[0]" = _embedding_bag_4[3];  _embedding_bag_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_5: "i64[2048]" = torch.ops.aten.select.int(primals_6, 0, 5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_5 = torch.ops.aten._embedding_bag.default(primals_17, primals_18, select_5, False, 0, True);  primals_17 = None
        getitem_20: "f32[2048, 64]" = _embedding_bag_5[0]
        getitem_21: "i64[204784]" = _embedding_bag_5[1]
        getitem_22: "i64[2048]" = _embedding_bag_5[2]
        getitem_23: "i64[0]" = _embedding_bag_5[3];  _embedding_bag_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_6: "i64[2048]" = torch.ops.aten.select.int(primals_6, 0, 6)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_6 = torch.ops.aten._embedding_bag.default(primals_19, primals_20, select_6, False, 0, True);  primals_19 = None
        getitem_24: "f32[2048, 64]" = _embedding_bag_6[0]
        getitem_25: "i64[204786]" = _embedding_bag_6[1]
        getitem_26: "i64[2048]" = _embedding_bag_6[2]
        getitem_27: "i64[0]" = _embedding_bag_6[3];  _embedding_bag_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_7: "i64[2048]" = torch.ops.aten.select.int(primals_6, 0, 7);  primals_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_7 = torch.ops.aten._embedding_bag.default(primals_21, primals_22, select_7, False, 0, True);  primals_21 = None
        getitem_28: "f32[2048, 64]" = _embedding_bag_7[0]
        getitem_29: "i64[204792]" = _embedding_bag_7[1]
        getitem_30: "i64[2048]" = _embedding_bag_7[2]
        getitem_31: "i64[0]" = _embedding_bag_7[3];  _embedding_bag_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:316 in interact_features, code: T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        cat: "f32[2048, 576]" = torch.ops.aten.cat.default([relu_1, getitem, getitem_4, getitem_8, getitem_12, getitem_16, getitem_20, getitem_24, getitem_28], 1);  getitem = getitem_4 = getitem_8 = getitem_12 = getitem_16 = getitem_20 = getitem_24 = getitem_28 = None
        view: "f32[2048, 9, 64]" = torch.ops.aten.reshape.default(cat, [2048, -1, 64]);  cat = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:318 in interact_features, code: Z = torch.bmm(T, torch.transpose(T, 1, 2))
        permute_2: "f32[2048, 64, 9]" = torch.ops.aten.permute.default(view, [0, 2, 1])
        bmm: "f32[2048, 9, 9]" = torch.ops.aten.bmm.default(view, permute_2);  view = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:329 in interact_features, code: li = torch.tensor(
        _tensor_constant0: "i64[36]" = self._tensor_constant0
        lift_fresh_copy: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:332 in interact_features, code: lj = torch.tensor(
        _tensor_constant1: "i64[36]" = self._tensor_constant1
        lift_fresh_copy_1: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:335 in interact_features, code: Zflat = Z[:, li, lj]
        index: "f32[2048, 36]" = torch.ops.aten.index.Tensor(bmm, [None, lift_fresh_copy, lift_fresh_copy_1]);  bmm = lift_fresh_copy = lift_fresh_copy_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:337 in interact_features, code: R = torch.cat([x] + [Zflat], dim=1)
        cat_1: "f32[2048, 100]" = torch.ops.aten.cat.default([relu_1, index], 1);  index = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        permute_3: "f32[100, 1024]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        addmm_2: "f32[2048, 1024]" = torch.ops.aten.addmm.default(primals_24, cat_1, permute_3);  primals_24 = permute_3 = None
        relu_2: "f32[2048, 1024]" = torch.ops.aten.relu.default(addmm_2);  addmm_2 = None
        permute_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_25, [1, 0])
        addmm_3: "f32[2048, 1024]" = torch.ops.aten.addmm.default(primals_26, relu_2, permute_4);  primals_26 = permute_4 = None
        relu_3: "f32[2048, 1024]" = torch.ops.aten.relu.default(addmm_3);  addmm_3 = None
        permute_5: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_4: "f32[2048, 1024]" = torch.ops.aten.addmm.default(primals_28, relu_3, permute_5);  primals_28 = permute_5 = None
        relu_4: "f32[2048, 1024]" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None
        permute_6: "f32[1024, 1]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        addmm_5: "f32[2048, 1]" = torch.ops.aten.addmm.default(primals_30, relu_4, permute_6);  primals_30 = permute_6 = None
        relu_5: "f32[2048, 1]" = torch.ops.aten.relu.default(addmm_5);  addmm_5 = None
        le: "b8[2048, 1]" = torch.ops.aten.le.Scalar(relu_5, 0)
        le_4: "b8[2048, 64]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        return (relu_5, primals_3, primals_4, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_27, primals_29, relu, select, getitem_1, getitem_2, getitem_3, select_1, getitem_5, getitem_6, getitem_7, select_2, getitem_9, getitem_10, getitem_11, select_3, getitem_13, getitem_14, getitem_15, select_4, getitem_17, getitem_18, getitem_19, select_5, getitem_21, getitem_22, getitem_23, select_6, getitem_25, getitem_26, getitem_27, select_7, getitem_29, getitem_30, getitem_31, permute_2, cat_1, relu_2, relu_3, relu_4, le, le_4)
