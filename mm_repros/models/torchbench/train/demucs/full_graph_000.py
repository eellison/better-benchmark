class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 5, 2, 426888][4268880, 853776, 426888, 1]cuda:0"):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[4][1]cuda:0" = torch.ops.prims.inductor_seeds.default(4, device(type='cuda', index=0))

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:105 in forward, code: th.rand(groups, group_size, streams, 1, 1, device=device), dim=1
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default: "f32[1, 4, 4, 1, 1][16, 4, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_random.default([1, 4, 4, 1, 1], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:104 in forward, code: permutations = th.argsort(
        sort = torch.ops.aten.sort.default(inductor_random_default, 1);  inductor_random_default = None
        getitem_1: "i64[1, 4, 4, 1, 1][16, 4, 1, 1, 1]cuda:0" = sort[1];  sort = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/__init__.py:29 in forward, code: sources = streams[:, 1:]
        slice_1: "f32[4, 4, 2, 426888][4268880, 853776, 426888, 1]cuda:0" = torch.ops.aten.slice.Tensor(arg0_1, 1, 1, 9223372036854775807);  arg0_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:65 in forward, code: signs = th.randint(
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_randint_default_2: "i64[4, 4, 1, 1][4, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_randint.default(0, 2, [4, 4, 1, 1], inductor_lookup_seed_default);  inductor_lookup_seed_default = None
        convert_element_type_default: "f32[4, 4, 1, 1][4, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_randint_default_2, torch.float32);  inductor_randint_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:68 in forward, code: wav = wav * (2 * signs - 1)
        mul: "f32[4, 4, 1, 1][4, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2);  convert_element_type_default = None
        sub: "f32[4, 4, 1, 1][4, 1, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
        mul_1: "f32[4, 4, 2, 426888][3415104, 853776, 426888, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_1, sub);  slice_1 = sub = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:48 in forward, code: left = th.randint(
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_randint_default_1: "i64[4, 4, 1, 1][4, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_randint.default(0, 2, [4, 4, 1, 1], inductor_lookup_seed_default_1);  inductor_lookup_seed_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:51 in forward, code: left = left.expand(-1, -1, -1, time)
        expand: "i64[4, 4, 1, 426888][4, 1, 1, 0]cuda:0" = torch.ops.aten.expand.default(inductor_randint_default_1, [-1, -1, -1, 426888]);  inductor_randint_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:53 in forward, code: wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        gather: "f32[4, 4, 1, 426888][1707552, 426888, 426888, 1]cuda:0" = torch.ops.aten.gather.default(mul_1, 2, expand)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:52 in forward, code: right = 1 - left
        sub_1: "i64[4, 4, 1, 426888][1707552, 426888, 426888, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, expand);  expand = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:53 in forward, code: wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        gather_1: "f32[4, 4, 1, 426888][1707552, 426888, 426888, 1]cuda:0" = torch.ops.aten.gather.default(mul_1, 2, sub_1);  mul_1 = sub_1 = None
        cat: "f32[4, 4, 2, 426888][3415104, 853776, 426888, 1]cuda:0" = torch.ops.aten.cat.default([gather, gather_1], 2);  gather = gather_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:35 in forward, code: indexes = th.arange(length, device=wav.device)
        iota: "i64[382788][1]cuda:0" = torch.ops.prims.iota.default(382788, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:28 in forward, code: offsets = th.randint(
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_randint_default: "i64[4, 4, 1, 1][4, 1, 1, 1]cuda:0" = torch.ops.prims.inductor_randint.default(0, 44100, [4, 4, 1, 1], inductor_lookup_seed_default_2);  inductor_lookup_seed_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:34 in forward, code: offsets = offsets.expand(-1, -1, channels, -1)
        expand_1: "i64[4, 4, 2, 1][4, 1, 0, 1]cuda:0" = torch.ops.aten.expand.default(inductor_randint_default, [-1, -1, 2, -1]);  inductor_randint_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:36 in forward, code: wav = wav.gather(3, indexes + offsets)
        add: "i64[4, 4, 2, 382788][3062304, 765576, 382788, 1]cuda:0" = torch.ops.aten.add.Tensor(iota, expand_1);  iota = expand_1 = None
        gather_2: "f32[4, 4, 2, 382788][3062304, 765576, 382788, 1]cuda:0" = torch.ops.aten.gather.default(cat, 3, add);  cat = add = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:103 in forward, code: wav = wav.view(groups, group_size, streams, channels, time)
        view: "f32[1, 4, 4, 2, 382788][12249216, 3062304, 765576, 382788, 1]cuda:0" = torch.ops.aten.reshape.default(gather_2, [1, 4, 4, 2, 382788]);  gather_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:107 in forward, code: wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
        expand_2: "i64[1, 4, 4, 2, 382788][16, 4, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_1, [-1, -1, -1, 2, 382788]);  getitem_1 = None
        gather_3: "f32[1, 4, 4, 2, 382788][12249216, 3062304, 765576, 382788, 1]cuda:0" = torch.ops.aten.gather.default(view, 1, expand_2);  view = expand_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/augment.py:108 in forward, code: wav = wav.view(batch, streams, channels, time)
        view_1: "f32[4, 4, 2, 382788][3062304, 765576, 382788, 1]cuda:0" = torch.ops.aten.reshape.default(gather_3, [4, 4, 2, 382788]);  gather_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/__init__.py:31 in forward, code: mix = sources.sum(dim=1)
        sum_1: "f32[4, 2, 382788][765576, 382788, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1, [1], dtype = torch.float32)
        return (view_1, sum_1)
