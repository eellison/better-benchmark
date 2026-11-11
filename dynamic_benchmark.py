"""Dynamic-family benchmark execution.

Live capture belongs to ``capture_hook.py``. This module consumes the captured
family contract and owns only four execution policies:

* one dynamic replay path, using backed tensor symbols when the capture has
  them and native ShapesSpec for genuinely unbacked symbols;
* partial specialization with ``--freeze``;
* full specialization with ``--static``;
* explicit compile/run history with ``--compile-at`` and ``--run-at``.

There are intentionally no alternate dynamic compilers here. One command means
one artifact model, so benchmark rows cannot silently compare different replay
mechanisms.
"""

import json
import time
import contextlib

import torch

import repro_harness as _harness
from input_codec import _symbol_point_value
from dynamic_shape_replay import (
    _distinct_dynamic_bindings,
    _dynamic_replay_args,
    _dynamic_replay_config,
    _effective_binding_for_row,
    _parse_bind_args,
    _parse_freeze_args,
    _recorded_point_bindings,
    _resolve_bound_configs,
    _prepare_dynamic_replay_for_repro,
    _symbols_and_guards_for_repro,
    format_binding,
)


def _dynamic_row_numerics(eager_fn, compiled_fn, make_inputs, label: str) -> dict:
    """The shared dynamic artifact against eager at one binding (one seed)."""
    from full_graph_harness import FullGraphNumericsError, run_numerics_gate

    try:
        summary, _ = run_numerics_gate(
            eager_fn, compiled_fn, make_inputs,
            label=label, variant="dynamic", seeds=1)
    except FullGraphNumericsError as e:
        summary = e.numerics
    return summary


def _run_bound_benchmark(repro_file, repro_cls, make_inputs_fn, parsed) -> dict:
    """Benchmark a dynamic family or an explicitly force-dynamic static repro."""
    symbols, _guards = _symbols_and_guards_for_repro(repro_file)
    is_family = bool(symbols)
    run_flag = getattr(parsed, "_run_at_flag", "--bind")
    compile_flag = getattr(parsed, "_compile_at_flag", "--prewarm")
    bindings = _parse_bind_args(parsed.bind, flag_name=run_flag)

    if parsed.static:
        mode = "static"
    elif parsed.dynamic or is_family:
        mode = "dynamic"
    else:
        mode = "static"

    frozen = {}
    if parsed.freeze:
        if not is_family:
            raise ValueError(
                "--freeze needs a dynamic repro with a symbols table")
        frozen = _parse_freeze_args(parsed.freeze, symbols)
        if set(frozen) == set(symbols):
            raise ValueError(
                "--freeze names every family symbol; use --static")
        if bindings:
            overlaid = []
            for binding in bindings:
                conflicts = sorted(
                    name for name, value in binding.items()
                    if name in frozen and frozen[name] != value
                )
                if conflicts:
                    raise ValueError(
                        f"{run_flag} binding {binding} contradicts "
                        f"--freeze {frozen} on {conflicts}")
                overlaid.append({**binding, **frozen})
            bindings = overlaid
        else:
            bindings = _recorded_point_bindings(
                repro_file, overlay=frozen)

    if parsed.static:
        if not is_family:
            raise ValueError(
                "--static requires a dynamic repro; this repro is already "
                "static and its plain run is the specialized benchmark")
        if not bindings:
            missing = sorted(
                name for name, definition in symbols.items()
                if not isinstance(
                    _symbol_point_value(definition), (int, float))
                or isinstance(
                    _symbol_point_value(definition), bool)
            )
            if missing:
                raise ValueError(
                    "--static needs an observed integer value for every "
                    f"symbol; no observed value for {missing}. Supply "
                    "explicit --run-at bindings.")
            bindings = [{
                name: _symbol_point_value(definition)
                for name, definition in symbols.items()
            }]

    if (is_family and mode == "dynamic" and not bindings
            and not _recorded_point_bindings(repro_file, overlay=frozen)):
        missing = sorted(
            name for name, definition in symbols.items()
            if name not in frozen
            and (
                not isinstance(
                    _symbol_point_value(definition), (int, float))
                or isinstance(_symbol_point_value(definition), bool)
            )
        )
        raise ValueError(
            "this dynamic family has no complete observed point; pass "
            f"explicit {run_flag} bindings for {missing}")

    if parsed.prewarm and mode == "static":
        raise ValueError(
            f"{compile_flag} has no effect in static mode; each binding "
            "gets a fresh specialized artifact")

    if is_family:
        for binding in bindings:
            missing = sorted(set(symbols) - set(binding))
            if missing:
                raise ValueError(
                    f"{run_flag} binding '{format_binding(binding)}' is "
                    f"incomplete: missing {missing}. Assign every symbol or "
                    "pin the remainder with --freeze.")

    rows = _resolve_bound_configs(
        repro_file, bindings, shape=parsed.shape)

    # An explicit binding describes one family point. All saved point labels
    # materialize identically at that binding, so measure one representative.
    seen_bindings = set()
    unique_rows = []
    for label, binding, config in rows:
        if binding is None:
            unique_rows.append((label, binding, config))
            continue
        key = tuple(sorted(binding.items()))
        if key not in seen_bindings:
            seen_bindings.add(key)
            unique_rows.append((label, binding, config))
    rows = unique_rows

    policies = {
        json.dumps(_harness.compile_policy_from_config(config), sort_keys=True)
        for _label, _binding, config in rows
    }
    if mode == "dynamic" and len(policies) > 1:
        raise ValueError(
            "one dynamic artifact cannot span input points with different "
            "compile policies")
    dynamic_policy = _harness.compile_policy_from_config(rows[0][2])

    def inputs_for(binding, config):
        inputs = _harness.make_inputs_from_config(config)
        if binding is None:
            inputs = _harness._merge_default_shape_params(
                inputs, _harness.make_inputs_safely(make_inputs_fn))
        return inputs

    def warm_bindings_for_family():
        if not parsed.prewarm:
            return (
                _distinct_dynamic_bindings(
                    repro_file, rows, n=2, frozen=frozen),
                "auto",
            )
        requested = _parse_bind_args(
            parsed.prewarm, flag_name=compile_flag)
        if frozen:
            overlaid = []
            for binding in requested:
                conflicts = sorted(
                    name for name, value in binding.items()
                    if name in frozen and frozen[name] != value
                )
                if conflicts:
                    raise ValueError(
                        f"{compile_flag} binding {binding} contradicts "
                        f"--freeze {frozen} on {conflicts}")
                overlaid.append({**binding, **frozen})
            requested = overlaid
        for binding in requested:
            missing = sorted(set(symbols) - set(binding))
            if missing:
                raise ValueError(
                    f"{compile_flag} binding "
                    f"'{format_binding(binding)}' is incomplete: missing "
                    f"{missing}")
        return requested, "explicit"

    compiled = None
    prepared_replay = None
    warm_bindings = []
    warm_source = "auto"

    def identity_artifact_args(inputs, _binding, _config):
        return inputs

    artifact_args = identity_artifact_args
    prewarmed = False

    if mode == "dynamic" and is_family:
        prepared_replay = _prepare_dynamic_replay_for_repro(
            repro_cls(), repro_file, frozen=frozen)
        shape_model = prepared_replay["module"]
        count_dynamic = (
            None if prepared_replay["kind"] == "backed" else False)

        def prepared_artifact_args(inputs, binding, config):
            effective = _effective_binding_for_row(binding, config)
            return _dynamic_replay_args(
                inputs, effective, prepared_replay)

        artifact_args = prepared_artifact_args

        warm_bindings, warm_source = warm_bindings_for_family()

        def shape_inputs(binding):
            config = next(iter(_harness.load_shape_configs(
                repro_file, symbol_bindings=binding).values()))
            inputs = inputs_for(binding, config)
            return artifact_args(inputs, binding, config)

        count_first = shape_inputs(warm_bindings[0])
        count_second = (
            shape_inputs(warm_bindings[1])
            if len(warm_bindings) > 1 else None
        )
        with _dynamic_replay_config(prepared_replay):
            n_kernels, kernel_names = _harness.count_kernels(
                shape_model,
                count_first,
                dynamic=count_dynamic,
                compile_policy=dynamic_policy,
                second_inputs=count_second,
            )
        torch._dynamo.reset()
        if not parsed.count_kernels_only:
            with _dynamic_replay_config(prepared_replay):
                compiled = _harness.compile_repro(
                    shape_model, compile_policy=dynamic_policy)
                with torch.no_grad():
                    for binding in warm_bindings:
                        compiled(*shape_inputs(binding))
                    torch.cuda.synchronize()
            prewarmed = True

    elif mode == "dynamic":
        # Compatibility for an explicitly forced symbol-less repro. There is
        # no captured family contract to reconstruct, so preserve the original
        # blanket dynamic=True behavior. Compile-history bindings have no
        # symbols to address and therefore cannot steer this fallback.
        if parsed.prewarm:
            print(
                f"  WARNING: {compile_flag} ignored for a symbol-less repro; "
                "blanket dynamic=True generalizes on its first invocation")
        first_label, first_binding, first_config = rows[0]
        del first_label
        first_inputs = inputs_for(first_binding, first_config)
        n_kernels, kernel_names = _harness.count_kernels(
            repro_cls(), first_inputs, dynamic=True,
            compile_policy=dynamic_policy)
        torch._dynamo.reset()
        shape_model = repro_cls()
        if not parsed.count_kernels_only:
            compiled = _harness.compile_repro(
                shape_model, dynamic=True, compile_policy=dynamic_policy)

    all_results = {}
    first_dynamic_row = True
    for label, binding, config in rows:
        binding_key = format_binding(binding)
        row_key = f"{label}::{binding_key}::{mode}"
        inputs = inputs_for(binding, config)
        measured_binding = (
            _effective_binding_for_row(binding, config)
            if is_family else binding
        )

        module = repro_cls()
        point_policy = _harness.compile_policy_from_config(config)
        with torch.no_grad():
            module(*inputs)

        graphs_before = _harness._unique_graph_count()
        if mode == "static":
            n_kernels, kernel_names = _harness.count_kernels(
                module, inputs, compile_policy=point_policy)
            if parsed.count_kernels_only:
                compiled_us = None
            else:
                torch._dynamo.reset()
                compiled_static = _harness.compile_repro(
                    module, compile_policy=point_policy)
                compiled_us = _harness.timed_min_us(
                    lambda: compiled_static(*inputs),
                    warmup=parsed.n_warmup,
                    rep=parsed.n_rep,
                )
            recompiled = None
        elif parsed.count_kernels_only:
            compiled_us = None
            recompiled = None
        else:
            call_args = artifact_args(inputs, binding, config)
            # Compile-history warmups run under no_grad. Keep measurement in
            # the same mode; changing grad mode is itself a Dynamo guard and
            # would create a second graph on the first timed point.
            replay_context = (
                _dynamic_replay_config(prepared_replay)
                if prepared_replay is not None
                else contextlib.nullcontext()
            )
            with replay_context, torch.no_grad():
                compiled_us = _harness.timed_min_us(
                    lambda: compiled(*call_args),
                    warmup=parsed.n_warmup,
                    rep=parsed.n_rep,
                )
            new_graph = (
                _harness._unique_graph_count() > graphs_before)
            if prewarmed:
                recompiled = new_graph
            elif first_dynamic_row:
                recompiled = False
            else:
                recompiled = new_graph
            if recompiled:
                print(
                    f"[{row_key}] WARNING: dynamic artifact recompiled at "
                    f"{format_binding(measured_binding)}; number may be "
                    "off-artifact")
            first_dynamic_row = False
            with (_dynamic_replay_config(prepared_replay)
                  if prepared_replay is not None
                  else contextlib.nullcontext()):
                numerics = _dynamic_row_numerics(
                    shape_model, compiled,
                    lambda: artifact_args(
                        inputs_for(binding, config), binding, config),
                    row_key)
            print(f"[{row_key}] numerics={numerics['status']}")

        if parsed.count_kernels_only:
            print(
                f"[{row_key}] binding={format_binding(measured_binding)} "
                f"mode={mode} kernels={n_kernels} (count only)")
        else:
            print(
                f"[{row_key}] binding={format_binding(measured_binding)} "
                f"mode={mode} time={compiled_us:8.1f} us"
                + (f" kernels={n_kernels}"
                   if n_kernels is not None else "")
                + (" RECOMPILED" if recompiled else ""))

        result = {
            "label": label,
            "binding": measured_binding,
            "mode": mode,
            "compile_policy": point_policy,
            "n_kernels": n_kernels,
            "kernel_names": kernel_names,
            "recompiled": recompiled,
        }
        if not parsed.count_kernels_only:
            result["compiled_us"] = compiled_us
        if mode == "dynamic" and not parsed.count_kernels_only:
            result["numerics"] = numerics
        if mode == "dynamic":
            result["compile_bindings"] = warm_bindings
            result["compile_bindings_source"] = warm_source
            result["frozen_symbols"] = frozen
        all_results[row_key] = result

    if parsed.output:
        with open(parsed.output, "w") as output:
            json.dump(all_results, output, indent=2)

    if parsed.update_perf:
        hardware = parsed.hardware or _harness._detect_hardware()
        for row_key, result in all_results.items():
            perf_entry = {
                key: value for key, value in result.items()
                if key != "kernel_names"
            }
            perf_entry["timestamp"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S")
            _harness._save_perf(
                repro_file, hardware, row_key, perf_entry)
        print(f"\n[perf] Saved to perf.json under hardware={hardware}")

    return all_results


def check_family(repro_file, repro_cls, bindings=None, device: str = "cuda") -> list:
    """Compile a dynamic repro's replay module once and run it at every
    binding (default: the recorded points), checking each against eager.

    No timing, so it runs on any device (``device`` overrides where the
    inputs are allocated). Returns one row per binding with the numerics
    summary and whether that binding compiled a new graph after the first.
    """
    symbols, _guards = _symbols_and_guards_for_repro(repro_file)
    if not symbols:
        raise ValueError(f"{repro_file} has no symbols table")
    prepared = _prepare_dynamic_replay_for_repro(repro_cls(), repro_file)
    model = prepared["module"]
    if bindings is None:
        bindings = _recorded_point_bindings(repro_file)

    def make_args(binding):
        config = next(iter(_harness.load_shape_configs(
            repro_file, symbol_bindings=binding).values()))
        config = dict(config, inputs=[
            dict(spec, device=device) if spec.get("kind", "tensor") == "tensor"
            else spec
            for spec in config["inputs"]])
        inputs = _harness.make_inputs_from_config(config)
        return _dynamic_replay_args(
            inputs, _effective_binding_for_row(binding, config), prepared)

    torch._dynamo.reset()
    rows = []
    with _dynamic_replay_config(prepared):
        compiled = torch.compile(model)
        for i, binding in enumerate(bindings):
            graphs_before = _harness._unique_graph_count()
            numerics = _dynamic_row_numerics(
                model, compiled, lambda: make_args(binding),
                format_binding(binding))
            rows.append({
                "binding": binding,
                "numerics": numerics["status"],
                "recompiled": (
                    i > 0 and _harness._unique_graph_count() > graphs_before),
            })
    return rows
