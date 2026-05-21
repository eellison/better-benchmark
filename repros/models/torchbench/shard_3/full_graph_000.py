class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 474][474, 1]cuda:0", arg1_1: "i64[2225][1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/embeddings/bert_embedding.py:440 in forward, code: word_mask = words.ne(self._word_pad_index)  # 为1的地方有word
        ne: "b8[4, 474][474, 1]cuda:0" = torch.ops.aten.ne.Scalar(arg0_1, 0)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/embeddings/bert_embedding.py:442 in forward, code: batch_word_pieces_length = self.word_pieces_lengths[words].masked_fill(word_mask.eq(False),
        eq: "b8[4, 474][474, 1]cuda:0" = torch.ops.aten.eq.Scalar(ne, False)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index: "i64[4, 474][474, 1]cuda:0" = torch.ops.aten.index.Tensor(arg1_1, [arg0_1]);  arg1_1 = arg0_1 = None
        where: "i64[4, 474][474, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, index);  eq = full_default = index = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/embeddings/bert_embedding.py:445 in forward, code: max_word_piece_length = batch_word_pieces_length.sum(dim=-1).max().item()  # 表示word piece的长度(包括padding)
        sum_3: "i64[4][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [-1])
        max_1: "i64[][]cuda:0" = torch.ops.aten.max.default(sum_3);  sum_3 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/embeddings/bert_embedding.py:441 in forward, code: seq_len = word_mask.sum(dim=-1)
        sum_1: "i64[4][1]cuda:0" = torch.ops.aten.sum.dim_IntList(ne, [-1])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/embeddings/bert_embedding.py:444 in forward, code: word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)  # batch_size
        sum_2: "i64[4][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [-1])
        return (max_1, ne, sum_1, where, sum_2)
