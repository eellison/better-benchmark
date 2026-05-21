class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 4096]", arg1_1: "f16[64, 1, 64]", arg2_1: "f16[1, 64, 192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:266 in forward, code: max_position_id = position_ids.max().item()
        max_1: "i64[]" = torch.ops.aten.max.default(arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:227 in forward, code: weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        expand: "f16[1, 64, 64, 64]" = torch.ops.aten.expand.default(arg1_1, [1, 64, 64, 64]);  arg1_1 = None
        expand_1: "f16[1, 64, 64, 192]" = torch.ops.aten.expand.default(arg2_1, [1, 64, 64, 192]);  arg2_1 = None
        return (max_1, expand, expand_1)
