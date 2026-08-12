class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8, 5, 2, 426888][4268880, 853776, 426888, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/__init__.py:29 in forward, code: sources = streams[:, 1:]
        slice_1: "bf16[8, 4, 2, 426888][4268880, 853776, 426888, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg0_1, 1, 1, 9223372036854775807);  arg0_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:26 in forward, code: wav = wav[..., :length]
        slice_2: "bf16[8, 4, 2, 382788][4268880, 853776, 426888, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 382788);  slice_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/__init__.py:31 in forward, code: mix = sources.sum(dim=1)
        sum_1: "bf16[8, 2, 382788][765576, 382788, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_2, [1])
        return (slice_2, sum_1)
