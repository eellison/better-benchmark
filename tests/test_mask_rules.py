import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import capture_hook
import mask_rules
from mask_rules import (
    approx_index_rule,
    jagged_rule,
    region_approx_index_rule,
    register_mask_ops,
    side_effect_rule,
    strip_side_effect_source,
)


@torch.library.custom_op("bbtest::jag_by_arg", mutates_args=())
def _jag_by_arg(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    return values.clone()


@_jag_by_arg.register_fake
def _(values, offsets):
    return torch.empty_like(values)


@torch.library.custom_op("bbtest::my_attn", mutates_args=())
def _my_attn(q: torch.Tensor, seq_offsets: torch.Tensor) -> torch.Tensor:
    return q.clone()


@_my_attn.register_fake
def _(q, seq_offsets):
    return torch.empty_like(q)


@torch.library.custom_op("bbtest::dense", mutates_args=())
def _dense(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y.sum()


@_dense.register_fake
def _(x, y):
    return torch.empty_like(x)


@torch.library.custom_op("bbtest::listed_hstu_fwd", mutates_args=())
def _listed_hstu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.clone()


@torch.library.custom_op("bbtest::listed_jag", mutates_args=())
def _listed_jag(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.clone()


@torch.library.custom_op("fbgemm::bbtest_jagged_to_padded_dense", mutates_args=())
def _fbgemm_jagged(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.clone()


@torch.library.custom_op("bbtest::window_state", mutates_args=("state",))
def _window_state(state: torch.Tensor, x: torch.Tensor) -> None:
    state.add_(x)


@torch.library.custom_op("bbtest::log", mutates_args=())
def _log(x: torch.Tensor) -> None:
    return None


@pytest.fixture
def registered_ops(monkeypatch):
    monkeypatch.setattr(mask_rules, "HSTU_OPS", list(mask_rules.HSTU_OPS))
    monkeypatch.setattr(mask_rules, "JAGGED_OPS", list(mask_rules.JAGGED_OPS))
    monkeypatch.setattr(mask_rules, "SIDE_EFFECT_OPS", dict(mask_rules.SIDE_EFFECT_OPS))
    monkeypatch.setattr(mask_rules, "HW_UNSUPPORTED_OPS", list(mask_rules.HW_UNSUPPORTED_OPS))
    register_mask_ops({
        "hstu": [{"namespace": "bbtest", "pattern": "listed_hstu"}],
        "jagged": [{"namespace": "bbtest", "pattern": "listed_jag"}],
        "side_effect": [
            {"namespace": "bbtest", "op": "window_state", "mutates": [0]},
            {"namespace": "bbtest", "op": "log"},
        ],
        "hw_unsupported": [
            {"namespace": "bbtest", "pattern": "^dense$", "min_capability": [10, 0]},
        ],
    })


def _graph(build):
    """GraphModule from build(graph, *placeholders) -> output value."""
    g = fx.Graph()
    phs = [g.placeholder(n) for n in ("x", "idx", "rep", "off")]
    g.output(build(g, *phs))
    return fx.GraphModule(torch.nn.Module(), g)


def _call(gm, target):
    nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target]
    assert len(nodes) == 1, nodes
    return nodes[0]


def _reason(node):
    return side_effect_rule(node) or jagged_rule(node)


@pytest.mark.parametrize("op, expected", [
    (torch.ops.bbtest.jag_by_arg.default, "jagged"),
    (torch.ops.bbtest.my_attn.default, "hstu"),
    (torch.ops.bbtest.dense.default, None),
    (torch.ops.bbtest.listed_hstu_fwd.default, "hstu"),
    (torch.ops.bbtest.listed_jag.default, "jagged"),
    (torch.ops.fbgemm.bbtest_jagged_to_padded_dense.default, "jagged"),
])
def test_jagged_rule_by_name_and_schema_arg(registered_ops, op, expected):
    gm = _graph(lambda g, x, idx, rep, off: g.call_function(op, (x, off)))
    assert jagged_rule(_call(gm, op)) == expected


def test_listed_ops_need_registration():
    op = torch.ops.bbtest.listed_hstu_fwd.default
    gm = _graph(lambda g, x, idx, rep, off: g.call_function(op, (x, off)))
    assert _reason(_call(gm, op)) is None


def test_rules_on_traced_graph():
    def f(x, off):
        return _jag_by_arg(x, off), _my_attn(x, off), _dense(x, x)

    gm = make_fx(f, tracing_mode="fake")(torch.randn(8, 4), torch.zeros(3, dtype=torch.long))
    got = {str(n.target): _reason(n) for n in gm.graph.nodes if n.op == "call_function"}
    assert got == {
        "bbtest.jag_by_arg.default": "jagged",
        "bbtest.my_attn.default": "hstu",
        "bbtest.dense.default": None,
    }


def test_repeat_interleave_tensor_is_jagged_scalar_repeat_is_not():
    gm = _graph(lambda g, x, idx, rep, off: (
        g.call_function(torch.ops.aten.repeat_interleave.Tensor, (rep,), {"output_size": 6}),
        g.call_function(torch.ops.aten.repeat_interleave.self_int, (x, 2)),
    ))
    assert jagged_rule(_call(gm, torch.ops.aten.repeat_interleave.Tensor)) == "jagged"
    assert jagged_rule(_call(gm, torch.ops.aten.repeat_interleave.self_int)) is None


def test_dense_aten_ops_are_not_masked():
    gm = _graph(lambda g, x, idx, rep, off: g.call_function(torch.ops.aten.mm.default, (x, x)))
    assert _reason(_call(gm, torch.ops.aten.mm.default)) is None


def test_side_effect_through_wrappers(registered_ops):
    window = torch.ops.bbtest.window_state.default
    log = torch.ops.bbtest.log.default
    afv2 = torch.ops.higher_order.auto_functionalized_v2
    with_effects = torch.ops.higher_order.with_effects
    gm = _graph(lambda g, x, idx, rep, off: (
        g.call_function(afv2, (window,), {"x": x, "_all_bases": [off]}),
        g.call_function(with_effects, (idx, log, x)),
        g.call_function(window, (off, x)),
    ))
    for target in (afv2, with_effects, window):
        assert side_effect_rule(_call(gm, target)) == "side_effect"
        assert _reason(_call(gm, target)) == "side_effect"


def test_wrapped_jagged_op_is_classified_by_inner_op(registered_ops):
    afv2 = torch.ops.higher_order.auto_functionalized_v2
    op = torch.ops.bbtest.listed_hstu_fwd.default
    gm = _graph(lambda g, x, idx, rep, off: g.call_function(afv2, (op,), {"x": x, "y": off}))
    assert jagged_rule(_call(gm, afv2)) == "hstu"


def test_index_from_input_is_approx_index():
    def build(g, x, idx, rep, off):
        shifted = g.call_function(torch.ops.aten.add.Tensor, (idx, 1))
        return (
            g.call_function(torch.ops.aten.index.Tensor, (x, [shifted])),
            g.call_function(torch.ops.aten.gather.default, (x, 0, idx)),
            g.call_function(torch.ops.aten.embedding.default, (x, idx)),
            g.call_function(torch.ops.aten.index_put.default, (x, [None, idx], x)),
        )

    gm = _graph(build)
    for target in (torch.ops.aten.index.Tensor, torch.ops.aten.gather.default,
                   torch.ops.aten.embedding.default, torch.ops.aten.index_put.default):
        assert approx_index_rule(_call(gm, target)) == "approx_index", target


def test_index_from_value_free_producers_is_exact():
    def build(g, x, idx, rep, off):
        n = g.call_function(torch.ops.aten.sym_size.int, (x, 0))
        ar = g.call_function(torch.ops.aten.arange.default, (n,))
        scaled = g.call_function(torch.ops.aten.mul.Tensor, (ar, 2))
        return g.call_function(torch.ops.aten.index.Tensor, (x, [scaled]))

    gm = _graph(build)
    assert _reason(_call(gm, torch.ops.aten.index.Tensor)) is None


def test_non_index_operand_from_input_is_not_approx():
    def build(g, x, idx, rep, off):
        ar = g.call_function(torch.ops.aten.arange.default, (4,))
        return g.call_function(torch.ops.aten.gather.default, (x, 0, ar))

    gm = _graph(build)
    assert approx_index_rule(_call(gm, torch.ops.aten.gather.default)) is None


def test_region_boundary_makes_index_synthetic():
    # The index comes from an arange, but a repro of the gather's partition
    # alone receives it as an input.
    def build(g, x, idx, rep, off):
        ar = g.call_function(torch.ops.aten.arange.default, (4,))
        shifted = g.call_function(torch.ops.aten.add.Tensor, (ar, 1))
        return g.call_function(torch.ops.aten.gather.default, (x, 0, shifted))

    gm = _graph(build)
    gather = _call(gm, torch.ops.aten.gather.default)
    shifted = _call(gm, torch.ops.aten.add.Tensor)
    arange = _call(gm, torch.ops.aten.arange.default)
    assert approx_index_rule(gather, region={gather}) == "approx_index"
    assert approx_index_rule(gather, region={gather, shifted, arange}) is None
    assert region_approx_index_rule([gather]) == "approx_index"
    assert region_approx_index_rule([arange, shifted, gather]) is None


def test_region_without_index_ops_is_unmasked():
    def build(g, x, idx, rep, off):
        a = g.call_function(torch.ops.aten.add.Tensor, (x, 1))
        return g.call_function(torch.ops.aten.relu.default, (a,))

    gm = _graph(build)
    assert region_approx_index_rule(
        [n for n in gm.graph.nodes if n.op == "call_function"]) is None


def test_scaffold_masks_through_registered_rules(registered_ops):
    assert approx_index_rule not in capture_hook.MASK_RULES
    log = torch.ops.bbtest.log.default
    attn = torch.ops.bbtest.my_attn.default
    gm = _graph(lambda g, x, idx, rep, off: (
        g.call_function(log, (x,)),
        g.call_function(attn, (x, off)),
        g.call_function(torch.ops.aten.repeat_interleave.Tensor, (rep,), {"output_size": 6}),
        g.call_function(torch.ops.aten.gather.default, (x, 0, idx)),
    ))
    assert capture_hook.mask_reasons(_call(gm, log)) == ("side_effect",)
    assert capture_hook.mask_reasons(_call(gm, attn)) == ("hstu",)
    ri = _call(gm, torch.ops.aten.repeat_interleave.Tensor)
    assert capture_hook.mask_reasons(ri) == ("jagged",)
    assert not capture_hook.partition_node_is_supported(ri)
    gather = _call(gm, torch.ops.aten.gather.default)
    assert capture_hook.mask_reasons(gather) == ()
    assert capture_hook.region_mask_reason([gather]) == "approx_index"
    assert capture_hook.graph_mask(gm) == {
        "reasons": ["hstu", "jagged", "side_effect"], "scope": "graph"}


def test_nested_only_mask_reaches_node_accounting():
    jag = _sub(lambda g, a: g.call_function(
        torch.ops.aten.repeat_interleave.Tensor, (a,), {"output_size": 6}))
    gm, _ = _with_subgraphs(_dense_sub(), jag)
    assert capture_hook.graph_mask(gm) == {"reasons": ["jagged"], "scope": "graph"}
    acct = capture_hook.graph_node_accounting(gm, components=[])
    assert acct["ops"]["masked"] == {
        "torch.ops.higher_order.cond": {"count": 1, "reasons": ["jagged"]}}


def _forward(body):
    return (
        "class Repro(torch.nn.Module):\n"
        "    def forward(self, arg0_1, arg1_1, arg2_1):\n"
        + textwrap.indent(textwrap.dedent(body).strip("\n"), " " * 8)
        + "\n"
    )


# The two forms a side-effect op takes in captured graphs: functionalized
# (runnable) and re-inplaced (post-grad).
_FUNCTIONALIZED = _forward("""
    sum_1 = torch.ops.aten.sum.default(arg1_1)
    auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.bbtest.window_state.default, x = sum_1, _state_base_index = 0, _all_bases = [arg0_1]);  sum_1 = None
    getitem_1 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
    mul = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
    return (mul,)
""")
_INPLACE = _forward("""
    sum_1: "f32[]" = torch.ops.aten.sum.default(arg1_1)
    window_state_default = torch.ops.bbtest.window_state.default(arg0_1, sum_1);  sum_1 = window_state_default = None
    mul: "f32[4]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    copy_: "f32[4]" = torch.ops.aten.copy_.default(arg0_1, arg0_1);  arg0_1 = copy_ = None
    return (mul,)
""")


_LOGGING = _forward("""
    add = torch.ops.aten.add.Tensor(arg1_1, 1)
    log = torch.ops.bbtest.log.default(add, 0.5, 'tag');  log = None
    return (add,)
""")


@pytest.mark.parametrize("source, label, kept, gone", [
    (_FUNCTIONALIZED, "bbtest.window_state", None, "copy_.default(arg0_1, getitem_1)"),
    (_INPLACE, "bbtest.window_state", "copy_.default(arg0_1, arg0_1)", None),
    (_LOGGING, "bbtest.log", "pass;  log = None", None),
])
def test_strip_side_effect_forms(registered_ops, source, label, kept, gone):
    new, stripped, blocked = strip_side_effect_source(source)
    assert (stripped, blocked) == ([label], [])
    assert "bbtest" not in new
    assert new.count("\n") == source.count("\n")
    compile(new, "<stripped>", "exec")
    assert kept is None or kept in new
    assert gone is None or gone not in new


@pytest.mark.parametrize("body, reason", [
    ("""
        log = torch.ops.bbtest.log.default(arg1_1)
        return (log,)
    """, "result read by compute"),
    ("""
        window_state_default = torch.ops.bbtest.window_state.default(arg0_1, arg1_1)
        mul = torch.ops.aten.mul.Tensor(arg0_1, 2)
        return (mul,)
    """, "mutated argument read by compute"),
    ("""
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.bbtest.window_state.default, x = arg1_1, _state_base_index = 0, _all_bases = [arg0_1])
        getitem_1 = auto_functionalized_v2[1]
        mul = torch.ops.aten.mul.Tensor(getitem_1, 2)
        return (mul,)
    """, "result read by compute"),
    ("""
        with_effects = torch.ops.higher_order.with_effects(arg2_1, torch.ops.bbtest.log.default, arg1_1)
        getitem = with_effects[0]
        return (getitem,)
    """, "with_effects token threading"),
])
def test_strip_blocked_when_results_are_used(registered_ops, body, reason):
    source = _forward(body)
    new, stripped, blocked = strip_side_effect_source(source)
    assert new == source
    assert stripped == []
    assert [r for _, r in blocked] == [reason]


def test_strip_unknown_mutation_is_blocked(registered_ops, monkeypatch):
    monkeypatch.setitem(mask_rules.SIDE_EFFECT_OPS, ("bbtest", "unregistered_op"), None)
    monkeypatch.setattr(mask_rules, "_mutated_positions", lambda key: None)
    source = _forward("""
        r = torch.ops.bbtest.unregistered_op.default(arg0_1);  r = None
        return (arg1_1,)
    """)
    new, stripped, blocked = strip_side_effect_source(source)
    assert new == source
    assert blocked == [("bbtest.unregistered_op", "unknown mutated arguments")]


def test_strip_is_a_noop_without_registered_ops():
    assert strip_side_effect_source(_INPLACE) == (_INPLACE, [], [])


def _sub(build):
    g = fx.Graph()
    a = g.placeholder("a")
    g.output(build(g, a))
    return fx.GraphModule(torch.nn.Module(), g)


def _with_subgraphs(true_gm, false_gm):
    """Root graph calling cond(pred, true_gm, false_gm, (rep,))."""
    root = torch.nn.Module()
    root.true_graph_0 = true_gm
    root.false_graph_0 = false_gm
    g = fx.Graph()
    pred = g.placeholder("pred")
    rep = g.placeholder("rep")
    t = g.get_attr("true_graph_0")
    f = g.get_attr("false_graph_0")
    out = g.call_function(torch.ops.higher_order.cond, (pred, t, f, (rep,)))
    g.output(out)
    gm = fx.GraphModule(root, g)
    return gm, _call(gm, torch.ops.higher_order.cond)


def _dense_sub():
    return _sub(lambda g, a: g.call_function(torch.ops.aten.relu.default, (a,)))


def test_rules_recurse_into_control_flow_subgraphs(registered_ops, monkeypatch):
    monkeypatch.setenv(mask_rules.TARGET_CAPABILITY_ENV, "9.0")
    jag = _sub(lambda g, a: g.call_function(
        torch.ops.aten.repeat_interleave.Tensor, (a,), {"output_size": 6}))
    _, cond = _with_subgraphs(_dense_sub(), jag)
    assert jagged_rule(cond) == "jagged"

    log = _sub(lambda g, a: g.call_function(torch.ops.bbtest.log.default, (a,)))
    _, cond = _with_subgraphs(log, _dense_sub())
    assert side_effect_rule(cond) == "side_effect"

    dense = _sub(lambda g, a: g.call_function(torch.ops.bbtest.dense.default, (a, a)))
    _, cond = _with_subgraphs(dense, _dense_sub())
    assert mask_rules.hw_unsupported_rule(cond) == "hw_unsupported"

    _, cond = _with_subgraphs(_dense_sub(), _dense_sub())
    assert _reason(cond) is None

    # Nested: invoke_subgraph -> cond -> repeat_interleave.
    inner, _ = _with_subgraphs(jag, _dense_sub())
    middle = torch.nn.Module()
    middle.subgraph_0 = inner
    g = fx.Graph()
    a = g.placeholder("a")
    sub = g.get_attr("subgraph_0")
    g.output(g.call_function(torch.ops.higher_order.invoke_subgraph, (sub, "id_0", a)))
    outer = fx.GraphModule(middle, g)
    node = _call(outer, torch.ops.higher_order.invoke_subgraph)
    assert jagged_rule(node) == "jagged"


def test_unknown_rule_reason_warns_once(monkeypatch, capsys):
    def typo_rule(node):
        return "Jagged"

    monkeypatch.setattr(capture_hook, "MASK_RULES", [typo_rule])
    monkeypatch.setattr(capture_hook, "REGION_MASK_RULES", [typo_rule])
    monkeypatch.setattr(capture_hook, "_warned_mask_reasons", set())
    gm = _graph(lambda g, x, idx, rep, off: g.call_function(torch.ops.aten.relu.default, (x,)))
    node = _call(gm, torch.ops.aten.relu.default)

    assert capture_hook.mask_reasons(node) == ()
    assert capture_hook.mask_reasons(node) == ()
    assert capture_hook.region_mask_reason([node]) is None
    err = capsys.readouterr().err
    assert err.count("typo_rule returned 'Jagged'") == 1


def test_strip_record_is_persisted_in_full_graph_sidecar(tmp_path):
    import json

    gm = make_fx(lambda x: x.relu() + 1)(torch.randn(4, 8))
    gm.meta["better_benchmark_side_effects"] = {
        "stripped": ["bbtest.window_state", "bbtest.window_state", "bbtest.log"],
        "blocked": [["bbtest.log", "result read by compute"]],
    }
    state = capture_hook._CaptureState(
        str(tmp_path / "capture"), label="m", graph_dir=str(tmp_path / "graphs"))
    path = state.capture_full_graph(gm, validate_partitions=False)

    meta = json.loads(Path(path).with_suffix(".meta.json").read_text())
    assert meta["side_effects"] == {
        "stripped": {"bbtest.window_state": 2, "bbtest.log": 1},
        "blocked": [["bbtest.log", "result read by compute"]],
    }


@pytest.mark.parametrize("capability, expected", [
    ("9.0", "hw_unsupported"),
    ("10.0", None),
    ("12.1", None),
])
def test_hw_unsupported_by_target_capability(registered_ops, monkeypatch, capability, expected):
    monkeypatch.setenv(mask_rules.TARGET_CAPABILITY_ENV, capability)
    op = torch.ops.bbtest.dense.default
    gm = _graph(lambda g, x, idx, rep, off: g.call_function(op, (x, x)))
    node = _call(gm, op)
    assert mask_rules.hw_unsupported_rule(node) == expected
    assert capture_hook.mask_reasons(node) == ((expected,) if expected else ())


def test_hw_unsupported_when_target_unknown(registered_ops, monkeypatch):
    monkeypatch.setattr(mask_rules, "target_capability", lambda: None)
    op = torch.ops.bbtest.dense.default
    gm = _graph(lambda g, x, idx, rep, off: g.call_function(op, (x, x)))
    assert mask_rules.hw_unsupported_rule(_call(gm, op)) == "hw_unsupported"


def test_unlisted_ops_never_hw_unsupported(monkeypatch):
    monkeypatch.setenv(mask_rules.TARGET_CAPABILITY_ENV, "7.0")
    gm = _graph(lambda g, x, idx, rep, off: g.call_function(torch.ops.aten.mm.default, (x, x)))
    assert mask_rules.hw_unsupported_rule(_call(gm, torch.ops.aten.mm.default)) is None
