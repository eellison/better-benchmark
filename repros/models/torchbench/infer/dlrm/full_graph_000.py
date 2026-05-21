class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[512, 512]", arg1_1: "f32[512]", arg2_1: "f32[2048, 512]", arg3_1: "f32[64, 512]", arg4_1: "f32[64]", arg5_1: "i64[8, 2048]", arg6_1: "f32[1000000, 64]", arg7_1: "i64[204790]", arg8_1: "f32[1000000, 64]", arg9_1: "i64[204789]", arg10_1: "f32[1000000, 64]", arg11_1: "i64[204793]", arg12_1: "f32[1000000, 64]", arg13_1: "i64[204790]", arg14_1: "f32[1000000, 64]", arg15_1: "i64[204793]", arg16_1: "f32[1000000, 64]", arg17_1: "i64[204784]", arg18_1: "f32[1000000, 64]", arg19_1: "i64[204786]", arg20_1: "f32[1000000, 64]", arg21_1: "i64[204792]", arg22_1: "f32[1024, 100]", arg23_1: "f32[1024]", arg24_1: "f32[1024, 1024]", arg25_1: "f32[1024]", arg26_1: "f32[1024, 1024]", arg27_1: "f32[1024]", arg28_1: "f32[1, 1024]", arg29_1: "f32[1]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 0)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag = torch.ops.aten._embedding_bag.default(arg6_1, arg7_1, select, False, 0, True);  arg6_1 = arg7_1 = select = None
        getitem: "f32[2048, 64]" = _embedding_bag[0];  _embedding_bag = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_1: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 1)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_1 = torch.ops.aten._embedding_bag.default(arg8_1, arg9_1, select_1, False, 0, True);  arg8_1 = arg9_1 = select_1 = None
        getitem_4: "f32[2048, 64]" = _embedding_bag_1[0];  _embedding_bag_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_2: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 2)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_2 = torch.ops.aten._embedding_bag.default(arg10_1, arg11_1, select_2, False, 0, True);  arg10_1 = arg11_1 = select_2 = None
        getitem_8: "f32[2048, 64]" = _embedding_bag_2[0];  _embedding_bag_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_3: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 3)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_3 = torch.ops.aten._embedding_bag.default(arg12_1, arg13_1, select_3, False, 0, True);  arg12_1 = arg13_1 = select_3 = None
        getitem_12: "f32[2048, 64]" = _embedding_bag_3[0];  _embedding_bag_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_4: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 4)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_4 = torch.ops.aten._embedding_bag.default(arg14_1, arg15_1, select_4, False, 0, True);  arg14_1 = arg15_1 = select_4 = None
        getitem_16: "f32[2048, 64]" = _embedding_bag_4[0];  _embedding_bag_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_5: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 5)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_5 = torch.ops.aten._embedding_bag.default(arg16_1, arg17_1, select_5, False, 0, True);  arg16_1 = arg17_1 = select_5 = None
        getitem_20: "f32[2048, 64]" = _embedding_bag_5[0];  _embedding_bag_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_6: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 6)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_6 = torch.ops.aten._embedding_bag.default(arg18_1, arg19_1, select_6, False, 0, True);  arg18_1 = arg19_1 = select_6 = None
        getitem_24: "f32[2048, 64]" = _embedding_bag_6[0];  _embedding_bag_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:298 in apply_emb, code: sparse_offset_group_batch = lS_o[k]
        select_7: "i64[2048]" = torch.ops.aten.select.int(arg5_1, 0, 7);  arg5_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:305 in apply_emb, code: V = E(sparse_index_group_batch, sparse_offset_group_batch)
        _embedding_bag_7 = torch.ops.aten._embedding_bag.default(arg20_1, arg21_1, select_7, False, 0, True);  arg20_1 = arg21_1 = select_7 = None
        getitem_28: "f32[2048, 64]" = _embedding_bag_7[0];  _embedding_bag_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:329 in interact_features, code: li = torch.tensor(
        _tensor_constant0: "i64[36]" = self._tensor_constant0

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:332 in interact_features, code: lj = torch.tensor(
        _tensor_constant1: "i64[36]" = self._tensor_constant1

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        permute: "f32[512, 512]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "f32[2048, 512]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        relu: "f32[2048, 512]" = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1: "f32[512, 64]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "f32[2048, 64]" = torch.ops.aten.addmm.default(arg4_1, relu, permute_1);  arg4_1 = relu = permute_1 = None
        relu_1: "f32[2048, 64]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:316 in interact_features, code: T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        cat: "f32[2048, 576]" = torch.ops.aten.cat.default([relu_1, getitem, getitem_4, getitem_8, getitem_12, getitem_16, getitem_20, getitem_24, getitem_28], 1);  getitem = getitem_4 = getitem_8 = getitem_12 = getitem_16 = getitem_20 = getitem_24 = getitem_28 = None
        view: "f32[2048, 9, 64]" = torch.ops.aten.reshape.default(cat, [2048, -1, 64]);  cat = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:318 in interact_features, code: Z = torch.bmm(T, torch.transpose(T, 1, 2))
        permute_2: "f32[2048, 64, 9]" = torch.ops.aten.permute.default(view, [0, 2, 1])
        bmm: "f32[2048, 9, 9]" = torch.ops.aten.bmm.default(view, permute_2);  view = permute_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:329 in interact_features, code: li = torch.tensor(
        lift_fresh_copy: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:332 in interact_features, code: lj = torch.tensor(
        lift_fresh_copy_1: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:335 in interact_features, code: Zflat = Z[:, li, lj]
        index: "f32[2048, 36]" = torch.ops.aten.index.Tensor(bmm, [None, lift_fresh_copy, lift_fresh_copy_1]);  bmm = lift_fresh_copy = lift_fresh_copy_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:337 in interact_features, code: R = torch.cat([x] + [Zflat], dim=1)
        cat_1: "f32[2048, 100]" = torch.ops.aten.cat.default([relu_1, index], 1);  relu_1 = index = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:284 in apply_mlp, code: return layers(x)
        permute_3: "f32[100, 1024]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_2: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg23_1, cat_1, permute_3);  arg23_1 = cat_1 = permute_3 = None
        relu_2: "f32[2048, 1024]" = torch.ops.aten.relu.default(addmm_2);  addmm_2 = None
        permute_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_3: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg25_1, relu_2, permute_4);  arg25_1 = relu_2 = permute_4 = None
        relu_3: "f32[2048, 1024]" = torch.ops.aten.relu.default(addmm_3);  addmm_3 = None
        permute_5: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_4: "f32[2048, 1024]" = torch.ops.aten.addmm.default(arg27_1, relu_3, permute_5);  arg27_1 = relu_3 = permute_5 = None
        relu_4: "f32[2048, 1024]" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None
        permute_6: "f32[1024, 1]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_5: "f32[2048, 1]" = torch.ops.aten.addmm.default(arg29_1, relu_4, permute_6);  arg29_1 = relu_4 = permute_6 = None
        relu_5: "f32[2048, 1]" = torch.ops.aten.relu.default(addmm_5);  addmm_5 = None
        return (relu_5,)
