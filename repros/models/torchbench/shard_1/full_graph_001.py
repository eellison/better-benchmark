class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "i64[2, 209981][209981, 1]cuda:0", primals_4: "f32[209981][1]cuda:0", convert_element_type_1: "f16[10000, 64][64, 1]cuda:0", permute_3: "f16[64, 64][64, 1]cuda:0", tangents_1: "f32[10000, 64][64, 1]cuda:0"):
        # File: /tmp/jenkins_pyg/tmpnymavaoh.py:247 in torch_dynamo_resume_in_forward_at_221, code: out = out + self.bias
        sum_1: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32)
        view_2: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [64]);  sum_1 = None

        # File: /tmp/jenkins_pyg/tmpnymavaoh.py:139 in _collect, code: edge_index_i = edge_def[i]
        select_1: "i64[209981][1]cuda:0" = torch.ops.aten.select.int(primals_3, 0, 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_1: "i64[209981, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(select_1, [-1, 1]);  select_1 = None
        expand: "i64[209981, 64][1, 0]cuda:0" = torch.ops.aten.expand.default(view_1, [209981, 64]);  view_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74 in scatter, code: return src.new_zeros(size).scatter_add_(dim, index, src)
        gather: "f32[209981, 64][64, 1]cuda:0" = torch.ops.aten.gather.default(tangents_1, 0, expand);  tangents_1 = expand = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/nn/conv/gcn_conv.py:241 in message, code: return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        view: "f32[209981, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_4, [-1, 1]);  primals_4 = None
        mul_1: "f32[209981, 64][64, 1]cuda:0" = torch.ops.aten.mul.Tensor(gather, view);  gather = view = None
        convert_element_type_4: "f16[209981, 64][64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1, torch.float16);  mul_1 = None

        # File: /tmp/jenkins_pyg/tmpnymavaoh.py:89 in _lift, code: return src.index_select(self.node_dim, index)
        full_default_1: "f16[10000, 64][64, 1]cuda:0" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/jenkins_pyg/tmpnymavaoh.py:88 in _lift, code: index = edge_index[dim]
        select: "i64[209981][1]cuda:0" = torch.ops.aten.select.int(primals_3, 0, 0);  primals_3 = None

        # File: /tmp/jenkins_pyg/tmpnymavaoh.py:89 in _lift, code: return src.index_select(self.node_dim, index)
        index_put: "f16[10000, 64][64, 1]cuda:0" = torch.ops.aten.index_put_.default(full_default_1, [select], convert_element_type_4, True);  full_default_1 = select = convert_element_type_4 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_1: "f16[64, 10000][1, 64]cuda:0" = torch.ops.aten.permute.default(index_put, [1, 0])
        mm_1: "f16[64, 64][64, 1]cuda:0" = torch.ops.aten.mm.default(permute_1, convert_element_type_1);  permute_1 = convert_element_type_1 = None
        mm_2: "f16[10000, 64][64, 1]cuda:0" = torch.ops.aten.mm.default(index_put, permute_3);  index_put = permute_3 = None
        convert_element_type_9: "f32[10000, 64][64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_10: "f32[64, 64][64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        return (convert_element_type_10, convert_element_type_9, None, None, view_2)
