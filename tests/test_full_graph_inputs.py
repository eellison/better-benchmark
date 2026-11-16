"""Saved full graphs as the harness loads them: integer inputs inferred from how the graph consumes them
(full_graph_inputs.py), and AMD-only Triton launch options dropped on CUDA."""

import textwrap

import torch

import full_graph_harness
import full_graph_inputs
from full_graph_harness import make_tensor_from_spec, parse_full_graph_inputs

# A graph in print_readable form: jagged lengths (arg0_1) and offsets (arg1_1, arg3_1) of a recommendation model,
# table ids (arg4_1), and a custom op this process may not have (fbgemm).
SRC = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i32[4][1]cuda:0", arg1_1: "i64[5][1]cuda:0", arg2_1: "f32[10, 8][8, 1]cuda:0", arg3_1: "i64[5][1]cuda:0", arg4_1: "i64[6][1]cuda:0", arg5_1: "f32[3, 16][16, 1]cuda:0", arg6_1: "i64[2][1]cuda:0"):
        convert_element_type: "i64[4][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg0_1, torch.int64);  arg0_1 = None
        repeat_interleave: "i64[10][1]cuda:0" = torch.ops.aten.repeat_interleave.Tensor(convert_element_type, output_size = 10);  convert_element_type = None
        jagged_to_padded_dense_forward: "f32[4, 7, 8][56, 8, 1]cuda:0" = torch.ops.fbgemm.jagged_to_padded_dense_forward.default(arg2_1, [arg1_1], [7])
        slice_1: "i64[4][1]cuda:0" = torch.ops.aten.slice.Tensor(arg3_1, 0, 1, 9223372036854775807)
        slice_2: "i64[4][1]cuda:0" = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, -1)
        sub: "i64[4][1]cuda:0" = torch.ops.aten.sub.Tensor(slice_1, slice_2);  slice_1 = slice_2 = None
        repeat_interleave_1: "i64[12][1]cuda:0" = torch.ops.aten.repeat_interleave.Tensor(sub, output_size = 12);  sub = None
        index: "f32[6, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(arg5_1, [arg4_1]);  arg5_1 = arg4_1 = None
        add: "i64[2][1]cuda:0" = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None
        return (repeat_interleave, jagged_to_padded_dense_forward, repeat_interleave_1, index, add)
'''  # noqa: B950


def _jagged(kind, total, max_len=None, **extra):
    return {"kind": kind, "total": total, **({"max": max_len} if max_len else {}), **extra}


def _unobserved_specs():
    specs = parse_full_graph_inputs(SRC)
    for spec in specs:
        if spec["dtype"].startswith("int"):
            spec["constraint_source"] = "default_unobserved"
    return specs

def test_graph_from_source_keeps_names_and_shapes():
    graph = full_graph_inputs.graph_from_source(SRC)
    nodes = {n.name: n for n in graph.nodes}
    assert list(nodes["jagged_to_padded_dense_forward"].meta["val"].shape) == [4, 7, 8]
    assert str(nodes["jagged_to_padded_dense_forward"].target) == "fbgemm.jagged_to_padded_dense_forward.default"
    assert nodes["repeat_interleave"].kwargs == {"output_size": 10}

def test_infers_jagged_lengths_and_offsets():
    gens = full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(SRC))
    assert gens == {
        "arg0_1": _jagged("lengths", 10),
        "arg1_1": _jagged("offsets", 10, max_len=7),
        "arg3_1": _jagged("offsets", 12),
    }

def test_refine_replaces_only_unconstrained_integer_inputs():
    specs = _unobserved_specs()
    refined = {s["name"]: s for s in full_graph_inputs.refine_unobserved_inputs(SRC, specs)}
    assert refined["arg0_1"]["gen"] == _jagged("lengths", 10)
    assert refined["arg4_1"]["gen"] == {"kind": "index", "low": 0, "high": 3}
    assert all(refined[n]["constraint_source"] == "graph_inference" for n in ("arg0_1", "arg1_1", "arg3_1", "arg4_1"))
    assert refined["arg2_1"] is specs[2]
    # the capture's own inference wins
    specs[4].update(constraint_source="graph_inference", gen={"kind": "index", "low": 0, "high": 2})
    assert full_graph_inputs.refine_unobserved_inputs(SRC, specs)[4] is specs[4]

def test_refine_overrides_observed_bounds_of_jagged_metadata():
    specs = _unobserved_specs()
    specs[1].update(constraint_source="observed", gen={"kind": "index", "low": 0, "high": 11})
    refined = {s["name"]: s for s in full_graph_inputs.refine_unobserved_inputs(SRC, specs)}
    assert refined["arg1_1"]["gen"] == _jagged("offsets", 10, max_len=7)


def test_refine_keeps_jagged_strided_and_aliased_inputs():
    specs = _unobserved_specs()
    specs[0].update(constraint_source="observed", gen=_jagged("lengths", 10))
    specs[1]["stride"] = [2]
    specs[3]["alias_group"] = 0
    refined = {s["name"]: s for s in full_graph_inputs.refine_unobserved_inputs(SRC, specs)}
    assert refined["arg0_1"] is specs[0]
    assert refined["arg1_1"]["gen"]["kind"] not in ("lengths", "offsets")
    assert refined["arg3_1"] is specs[3]


def test_refine_keeps_specs_when_the_analysis_fails(monkeypatch):
    def boom(graph):
        raise IndexError("an op form the analysis does not cover")

    monkeypatch.setattr(full_graph_inputs, "infer_jagged_inputs_from_gm", boom)
    specs = _unobserved_specs()
    assert full_graph_inputs.refine_unobserved_inputs(SRC, specs) is specs

def test_load_full_graph_generates_jagged_inputs(tmp_path):
    path = tmp_path / "full_graph_000.py"
    path.write_text(SRC.replace("cuda:0", "cpu"))
    _, inputs, definition = full_graph_harness.load_full_graph(path, default_device="cpu")
    assert definition.input_specs[0]["gen"] == _jagged("lengths", 10)
    assert inputs[0].dtype == torch.int32
    assert inputs[0].numel() == 4 and int(inputs[0].sum()) == 10 and int(inputs[0].min()) >= 0
    offsets = inputs[1]
    assert offsets[0] == 0 and offsets[-1] == 10 and 0 <= int(offsets.diff().min()) and int(offsets.diff().max()) <= 7
    assert int(inputs[4].max()) < 3


def test_key_bound_ops_bound_ids_of_registered_custom_ops(monkeypatch):
    src = textwrap.dedent('''
    class Repro(torch.nn.Module):
        def forward(self, arg0_1: "f32[6][1]cuda:0", arg1_1: "i64[6][1]cuda:0", arg2_1: "i64[2][1]cuda:0"):
            to_dense = torch.ops.ext.to_dense.default(arg0_1, arg1_1, arg2_1, 7);  arg0_1 = arg1_1 = arg2_1 = None
            return (to_dense,)
    ''')  # noqa: B950
    monkeypatch.setitem(full_graph_inputs.KEY_BOUND_OPS, "ext.to_dense.default", (1, 3))
    specs = parse_full_graph_inputs(src)
    specs[1]["constraint_source"] = "default_unobserved"
    assert full_graph_inputs.refine_unobserved_inputs(src, specs)[1]["gen"] == {"kind": "index", "low": 0, "high": 7}


def test_refine_keeps_specs_of_a_source_it_cannot_parse():
    specs = [{"name": "arg0_1", "kind": "tensor", "dtype": "int64", "shape": [3]}]
    assert full_graph_inputs.refine_unobserved_inputs("class Repro:\n    pass\n", specs) is specs


def test_segment_ends_from_cumsum_bound_lengths_by_the_indexed_extent():
    src = textwrap.dedent('''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[4][1]cuda:0", arg1_1: "i64[9][1]cuda:0"):
        cumsum: "i64[4][1]cuda:0" = torch.ops.aten.cumsum.default(arg0_1, 0);  arg0_1 = None
        sub: "i64[4][1]cuda:0" = torch.ops.aten.sub.Tensor(cumsum, 1);  cumsum = None
        index: "i64[4][1]cuda:0" = torch.ops.aten.index.Tensor(arg1_1, [sub]);  arg1_1 = sub = None
        return (index,)
''')
    gens = full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(src))
    assert gens == {"arg0_1": _jagged("lengths", 9)}
    # the last segment end itself as a row id, or a start (exclusive cumsum): at most 8
    for producer, extra in (("aten.cumsum.default(arg0_1, 0)", ""),
                            ("fbgemm.asynchronous_exclusive_cumsum.default(arg0_1)", "")):
        other = src.replace("torch.ops.aten.cumsum.default(arg0_1, 0)", f"torch.ops.{producer}").replace(
            "torch.ops.aten.sub.Tensor(cumsum, 1)", "torch.ops.aten.alias.default(cumsum)")
        gens = full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(other))
        assert gens == {"arg0_1": _jagged("lengths", 8)}, producer

def test_pack_segments_and_dense_to_jagged_pin_their_lengths_and_offsets():
    src = textwrap.dedent('''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[30, 8][8, 1]cuda:0", arg1_1: "i64[4][1]cuda:0", arg2_1: "f32[4, 6, 8][48, 8, 1]cuda:0", arg3_1: "i64[5][1]cuda:0"):
        pack_segments: "f32[4, 9, 8][72, 8, 1]cuda:0" = torch.ops.fbgemm.pack_segments.default(arg0_1, arg1_1, 9);  arg0_1 = arg1_1 = None
        dense_to_jagged = torch.ops.fbgemm.dense_to_jagged.default(arg2_1, [arg3_1], 20);  arg2_1 = arg3_1 = None
        getitem: "f32[20, 8][8, 1]cuda:0" = dense_to_jagged[0];  dense_to_jagged = None
        return (pack_segments, getitem)
''')  # noqa: B950
    gens = full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(src))
    assert gens == {"arg1_1": _jagged("lengths", 30, max_len=9), "arg3_1": _jagged("offsets", 20, max_len=6)}
    offsets = make_tensor_from_spec({"shape": [5], "dtype": "int64", "device": "cpu", "gen": gens["arg3_1"]})
    assert offsets.tolist() == [0, 5, 10, 15, 20]


def test_dynamic_graphs_keep_their_specs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(full_graph_inputs, "refine_unobserved_inputs", lambda content, specs: calls.append(1) or specs)
    path = tmp_path / "full_graph_000.py"
    path.write_text(SRC.replace("cuda:0", "cpu"))
    full_graph_harness.load_full_graph_definition(path)
    assert calls == [1]
    monkeypatch.setattr(full_graph_harness, "load_full_graph_sidecar", lambda p: {"symbols": {"s0": {"hint": 4}}})
    monkeypatch.setattr(full_graph_harness, "_validate_sidecar_inputs", lambda *a: None)
    monkeypatch.setattr("input_codec.validate_bindings", lambda *a: None)
    monkeypatch.setattr("input_codec.evaluate_spec", lambda spec, *a: spec)
    full_graph_harness.load_full_graph_definition(path)
    assert calls == [1]


def test_user_triton_indices_are_bounded_by_the_tensor_they_are_named_after():
    src = textwrap.dedent('''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[50, 8][8, 1]cuda:0", arg1_1: "i64[6][1]cuda:0", arg2_1: "f32[6, 8][8, 1]cuda:0"):
        triton_kernel_wrapper_mutation_proxy = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 0, grid = [(6, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'input_group_ptrs0': arg0_1, 'indices_group_ptrs0': arg1_1, 'single_output_ptr': arg2_1, 'D': 8});  arg0_1 = arg1_1 = triton_kernel_wrapper_mutation_proxy = None
        return (arg2_1,)
''')  # noqa: B950
    specs = parse_full_graph_inputs(src)
    specs[1]["constraint_source"] = "default_unobserved"
    refined = full_graph_inputs.refine_unobserved_inputs(src, specs)
    assert refined[1]["gen"] == {"kind": "index", "low": 0, "high": 50}

def test_user_triton_offsets_end_at_the_rows_of_the_tensor_they_split():
    src = textwrap.dedent('''
    class Repro(torch.nn.Module):
        def forward(self, arg0_1: "f32[70, 8][8, 1]cuda:0", arg1_1: "i64[5][1]cuda:0", arg2_1: "f32[16, 8][8, 1]cuda:0", arg3_1: "f32[54, 8][8, 1]cuda:0"):
            triton_kernel_wrapper_mutation_proxy = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 0, grid = [(1, 4, 1)], tma_descriptor_metadata = {}, kwargs = {'JaggedIn': arg0_1, 'OffsetsB': arg1_1, 'OutA': arg2_1, 'OutB': arg3_1});  arg0_1 = arg1_1 = triton_kernel_wrapper_mutation_proxy = None
            return (arg2_1, arg3_1)
    ''')  # noqa: B950
    gens = full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(src))
    assert gens == {"arg1_1": _jagged("offsets", 54)}


def test_a_bare_offsets_ptr_names_no_tensor():
    src = textwrap.dedent('''
    class Repro(torch.nn.Module):
        def forward(self, arg0_1: "f32[30, 8][8, 1]cuda:0", arg1_1: "i64[5][1]cuda:0", arg2_1: "f32[64, 8][8, 1]cuda:0"):
            triton_kernel_wrapper_mutation_proxy = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 0, grid = [(4, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'values_ptr': arg0_1, 'offsets_ptr': arg1_1, 'out_ptr': arg2_1, 'max_len': 16});  arg0_1 = arg1_1 = triton_kernel_wrapper_mutation_proxy = None
            return (arg2_1,)
    ''')  # noqa: B950
    assert full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(src)) == {}
    named = src.replace("'offsets_ptr'", "'values_offsets_ptr'")
    assert full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(named)) == {
        "arg1_1": _jagged("offsets", 30)}


def test_lengths_keep_their_total_over_an_infeasible_max():
    lengths = make_tensor_from_spec({"shape": [4], "dtype": "int64", "device": "cpu",
                                     "gen": _jagged("lengths", 100, max_len=10)})
    assert lengths.tolist() == [25, 25, 25, 25]


def test_user_triton_backward_indices_are_bounded_by_the_gradient_they_scatter_into():
    src = textwrap.dedent('''
    class Repro(torch.nn.Module):
        def forward(self, arg0_1: "i64[600][1]cuda:0", arg1_1: "f32[600, 8][8, 1]cuda:0"):
            full: "f32[36, 8][8, 1]cuda:0" = torch.ops.aten.full.default([36, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
            triton_kernel_wrapper_mutation_proxy = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 0, grid = [(6, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'indices_group_ptrs0': arg0_1, 'd_in_ptrs0': full, 'single_d_out_ptr': arg1_1, 'D': 8});  arg0_1 = arg1_1 = triton_kernel_wrapper_mutation_proxy = None
            return (full,)
    ''')  # noqa: B950
    specs = parse_full_graph_inputs(src)
    specs[0]["constraint_source"] = "default_unobserved"
    assert full_graph_inputs.refine_unobserved_inputs(src, specs)[0]["gen"] == {"kind": "index", "low": 0, "high": 36}


def test_bucketize_boundaries_are_sorted():
    src = textwrap.dedent('''
    class Repro(torch.nn.Module):
        def forward(self, arg0_1: "f32[64][1]cuda:0", arg1_1: "i64[9][1]cuda:0", arg2_1: "f32[1, 7][7, 1]cuda:0"):
            bucketize: "i64[64][1]cuda:0" = torch.ops.aten.bucketize.Tensor(arg0_1, arg1_1);  arg1_1 = None
            view: "f32[7][1]cuda:0" = torch.ops.aten.view.default(arg2_1, [7]);  arg2_1 = None
            bucketize_1: "i64[64][1]cuda:0" = torch.ops.aten.bucketize.Tensor(arg0_1, view);  arg0_1 = view = None
            return (bucketize, bucketize_1)
    ''')  # noqa: B950
    specs = parse_full_graph_inputs(src)
    specs[1]["constraint_source"] = "default_unobserved"
    refined = full_graph_inputs.refine_unobserved_inputs(src, specs)
    assert "gen" not in refined[0]
    assert refined[1]["gen"] == {"kind": "sorted", "low": 0, "high": 9}
    assert refined[2]["gen"] == {"kind": "sorted"}
    for spec in refined[1:]:
        values = make_tensor_from_spec({**spec, "device": "cpu"}).flatten()
        assert bool((values[1:] >= values[:-1]).all())


def test_offsets_read_by_searchsorted_keep_their_offsets_generator():
    src = textwrap.dedent('''
    class Repro(torch.nn.Module):
        def forward(self, arg0_1: "f32[6, 8][8, 1]cuda:0", arg1_1: "i64[11][1]cuda:0", arg2_1: "i64[6][1]cuda:0"):
            jagged_to_padded_dense_forward: "f32[10, 3, 8][24, 8, 1]cuda:0" = torch.ops.fbgemm.jagged_to_padded_dense_forward.default(arg0_1, [arg1_1], [3]);  arg0_1 = None
            searchsorted: "i64[6][1]cuda:0" = torch.ops.aten.searchsorted.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
            return (jagged_to_padded_dense_forward, searchsorted)
    ''')  # noqa: B950
    specs = parse_full_graph_inputs(src)
    specs[1]["constraint_source"] = "default_unobserved"
    assert full_graph_inputs.refine_unobserved_inputs(src, specs)[1]["gen"] == _jagged("offsets", 6, max_len=3)


def test_slices_of_a_reshaped_or_from_the_end_lengths_tensor():
    def graph(lengths, view, start, stop):
        return textwrap.dedent(f'''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[{lengths}][1]cuda:0"):
            {view}
            slice_1: "i64[4][1]cuda:0" = torch.ops.aten.slice.Tensor(src, 0, {start}, {stop})
            repeat_interleave: "i64[5][1]cuda:0" = torch.ops.aten.repeat_interleave.Tensor(slice_1, output_size = 5)
            return (repeat_interleave,)
''').replace("\n            ", "\n        ")

    def infer(src):
        return full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(src))

    # lengths[-4:]: the last four elements, not the first ones
    src = graph(12, 'src: "i64[12][1]cuda:0" = torch.ops.aten.alias.default(arg0_1)', -4, 9223372036854775807)
    assert infer(src) == {"arg0_1": _jagged("lengths", 5, segments=[[8, 12, 5]])}
    # a slice of a reshaped view does not index the flat tensor
    src = graph(16, 'src: "i64[4, 4][4, 1]cuda:0" = torch.ops.aten.view.default(arg0_1, [4, 4])', 0, 1)
    assert infer(src) == {}


def test_slices_of_a_lengths_tensor_get_their_own_totals():
    src = textwrap.dedent('''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i32[100][1]cuda:0", arg1_1: "f32[20, 1][1, 1]cuda:0"):
        slice_1: "i32[10][1]cuda:0" = torch.ops.aten.slice.Tensor(arg0_1, 0, 30, 40);  arg0_1 = None
        full: "i64[1][1]cuda:0" = torch.ops.aten.full.default([1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cumsum: "i64[10][1]cuda:0" = torch.ops.aten.cumsum.default(slice_1, 0);  slice_1 = None
        cat: "i64[11][1]cuda:0" = torch.ops.aten.cat.default([full, cumsum]);  full = cumsum = None
        jagged_to_padded_dense_forward: "f32[10, 3, 1][3, 1, 1]cuda:0" = torch.ops.fbgemm.jagged_to_padded_dense_forward.default(arg1_1, [cat], [3]);  arg1_1 = cat = None
        return (jagged_to_padded_dense_forward,)
''')  # noqa: B950
    gen = full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(src))["arg0_1"]
    assert gen == _jagged("lengths", 20, segments=[[30, 40, 20]])
    lengths = make_tensor_from_spec({"shape": [100], "dtype": "int32", "device": "cpu", "gen": gen})
    assert int(lengths.sum()) == int(lengths[30:40].sum()) == 20
    # the same slice taken twice counts once
    twice = src.replace("cumsum: ", "slice_2: \"i32[10][1]cuda:0\" = torch.ops.aten.slice.Tensor(arg0_1, 0, 30, 40)\n"
                        "        cumsum_2: \"i64[10][1]cuda:0\" = torch.ops.aten.cumsum.default(slice_2, 0)\n"
                        "        cat_2: \"i64[11][1]cuda:0\" = torch.ops.aten.cat.default([full, cumsum_2])\n"
                        "        dense_2: \"f32[10, 3, 1][3, 1, 1]cuda:0\" = torch.ops.fbgemm.jagged_to_padded_dense_forward.default(arg1_1, [cat_2], [3])\n"  # noqa: B950
                        "        cumsum: ", 1)
    gen = full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(twice))["arg0_1"]
    assert gen == _jagged("lengths", 20, segments=[[30, 40, 20]])
    # a whole-tensor total too: the rest of it goes to the largest gap; one the slices exceed leaves the input alone
    def whole(total):
        return src.replace("return (jagged_to_padded_dense_forward,)",
                           f"repeat_interleave: \"i64[{total}][1]cuda:0\" = torch.ops.aten.repeat_interleave.Tensor(arg0_1, output_size = {total})\n"  # noqa: B950
                           "        return (jagged_to_padded_dense_forward, repeat_interleave)")
    gen = full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(whole(50)))["arg0_1"]
    assert gen == _jagged("lengths", 50, segments=[[30, 40, 20], [40, 100, 30]])
    lengths = make_tensor_from_spec({"shape": [100], "dtype": "int32", "device": "cpu", "gen": gen})
    assert (int(lengths.sum()), int(lengths[30:40].sum())) == (50, 20)
    assert "arg0_1" not in full_graph_inputs.infer_jagged_inputs_from_gm(full_graph_inputs.graph_from_source(whole(10)))


# A launch the way a graph captured on MI300 has it, and an autotune config of the kernel.
_AMD_GRAPH = (
    "torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 1, grid = [(1, 1, 1)], "
    "tma_descriptor_metadata = {}, kwargs = {'Q': view_1, 'HEAD_DIM': 128, 'waves_per_eu': 2, "
    "'allow_flush_denorm': True, 'STAGE': 1}, launch_kwargs = ('HEAD_DIM', 'waves_per_eu', 'allow_flush_denorm', 'STAGE'))\n"
    "triton.Config({'BLOCK_M': 16, 'matrix_instr_nonkdim': 16, 'kpack': 2, 'waves_per_eu': 1}, num_warps=4)\n"
)


def test_drop_amd_launch_options_on_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    out = full_graph_harness.drop_amd_launch_options(_AMD_GRAPH)
    for option in ("waves_per_eu", "matrix_instr_nonkdim", "kpack", "allow_flush_denorm"):
        assert option not in out
    assert "kwargs = {'Q': view_1, 'HEAD_DIM': 128, 'STAGE': 1}" in out
    assert "launch_kwargs = ('HEAD_DIM', 'STAGE'))" in out
    assert "triton.Config({'BLOCK_M': 16, }, num_warps=4)" in out
    compile(out, "graph", "exec")
    one = full_graph_harness.drop_amd_launch_options("f(launch_kwargs = ('waves_per_eu', 'STAGE'), k = {'kpack' : 2})")
    assert one == "f(launch_kwargs = ('STAGE',), k = {})"
    assert full_graph_harness.drop_amd_launch_options("g(k = {'x': 'kpack', 'y': 1})") == "g(k = {'x': 'kpack', 'y': 1})"


def test_drop_amd_launch_options_keeps_them_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert full_graph_harness.drop_amd_launch_options(_AMD_GRAPH) == _AMD_GRAPH


def test_bench_counts_graph_inferred_jagged_inputs_as_safe():
    from scripts.bench_parallel import _is_graph_inferred_integer_input

    assert _is_graph_inferred_integer_input({"gen": _jagged("offsets", 10), "constraint_source": "graph_inference"})
