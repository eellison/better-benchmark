"""
Ingest post-grad FX graphs from CI tlparse artifacts.

Downloads tlparse artifacts from a GitHub Actions run, loads the
inductor_post_grad_graph_*.txt files as GraphModules, runs the capture
hook's partitioning on each, and merges the resulting regions into the
canonical repro set.

Usage:
    # Download and ingest from a specific run:
    python scripts/ingest_tlparse.py --run-id 25956360525 --suite torchbench

    # Ingest from already-downloaded tlparse dir:
    python scripts/ingest_tlparse.py --tlparse-dir /tmp/tb_tlparse3 --suite torchbench

    # Just list what's available in a run:
    python scripts/ingest_tlparse.py --run-id 25956360525 --list-only
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def download_tlparse(run_id: int, suite: str | None, output_dir: Path) -> Path:
    """Download tlparse artifacts from a GitHub Actions run."""
    pattern = "tlparse-*"
    if suite:
        pattern = f"tlparse-*{suite}*"

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gh", "run", "download", str(run_id),
        "--repo", "pytorch/pytorch",
        "--pattern", pattern,
        "--dir", str(output_dir),
    ]
    print(f"Downloading: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_dir


def find_post_grad_graphs(tlparse_dir: Path) -> list[Path]:
    """Find all inductor_post_grad_graph_*.txt files."""
    return sorted(tlparse_dir.rglob("inductor_post_grad_graph_*.txt"))


def find_fx_graph_runnables(tlparse_dir: Path) -> list[Path]:
    """Find all fx_graph_runnable_*.txt files."""
    return sorted(tlparse_dir.rglob("fx_graph_runnable_*.txt"))


def infer_model_name(graph_path: Path) -> str:
    """Infer model name from the tlparse directory structure."""
    # Path looks like: tlparse-...-test-inductor_torchbench_perf-4-6-.../  -_0_0_0/  file.txt
    parts = str(graph_path).split("/")
    for part in parts:
        if "tlparse-" in part:
            # Extract suite and model info from directory name
            # e.g. tlparse-runattempt1-test-inductor_torchbench_perf_cuda_h100-1-9-...
            return part.split("_76")[0].replace("tlparse-runattempt1-test-", "")
    return "unknown"


def _parse_input_shapes_from_reader(content: str):
    """Parse input tensor shapes/dtypes from reader.tensor() lines in load_args.

    Lines look like:
        reader.tensor(buf0, (32, 3, 3, 3), is_leaf=True)  # primals_1
        reader.tensor(buf2, (), dtype=torch.int64, is_leaf=True)  # primals_3
    """
    import re
    import torch

    dtype_map = {
        'torch.float32': torch.float32,
        'torch.float16': torch.float16,
        'torch.bfloat16': torch.bfloat16,
        'torch.int64': torch.int64,
        'torch.int32': torch.int32,
        'torch.int16': torch.int16,
        'torch.int8': torch.int8,
        'torch.uint8': torch.uint8,
        'torch.bool': torch.bool,
        'torch.float64': torch.float64,
        'torch.complex64': torch.complex64,
        'torch.complex128': torch.complex128,
    }

    inputs = []
    for m in re.finditer(
        r'reader\.tensor\(\w+,\s*\(([^)]*)\)(?:,\s*dtype=([^,)]+))?', content
    ):
        shape_str, dtype_str = m.group(1), m.group(2)
        # Handle symbolic shapes (e.g. 's28') by substituting a concrete value
        dims = []
        for x in shape_str.split(','):
            x = x.strip()
            if not x:
                continue
            try:
                dims.append(int(x))
            except ValueError:
                # Symbolic dim (e.g. 's28', 's0') - use 32 as reasonable batch size
                dims.append(32)
        shape = tuple(dims)
        dtype = dtype_map.get(dtype_str, torch.float32) if dtype_str else torch.float32
        inputs.append((shape, dtype))

    return inputs


def _parse_input_shapes_from_annotations(content: str):
    """Parse input tensor shapes/dtypes from forward signature annotations.

    Annotations look like:
        primals_1: "f32[768, 768][768, 1]cuda:0"
        primals_2: "i64[1, 512][512, 1]cuda:0"
    """
    import re
    import torch

    dtype_map = {
        'f32': torch.float32,
        'f16': torch.float16,
        'bf16': torch.bfloat16,
        'i64': torch.int64,
        'i32': torch.int32,
        'i16': torch.int16,
        'i8': torch.int8,
        'u8': torch.uint8,
        'b8': torch.bool,
        'f64': torch.float64,
        'c64': torch.complex64,
        'c128': torch.complex128,
    }

    # Match the forward signature
    fwd_match = re.search(r'def forward\(self,?\s*(.*?)\):', content, re.DOTALL)
    if not fwd_match:
        return []

    def split_signature_args(sig: str) -> list[str]:
        args = []
        start = 0
        in_quote = False
        quote = ""
        for i, ch in enumerate(sig):
            if ch in {'"', "'"}:
                if in_quote and ch == quote:
                    in_quote = False
                    quote = ""
                elif not in_quote:
                    in_quote = True
                    quote = ch
            elif ch == "," and not in_quote:
                part = sig[start:i].strip()
                if part:
                    args.append(part)
                start = i + 1
        part = sig[start:].strip()
        if part:
            args.append(part)
        return args

    def parse_symbol_hint(annotation: str) -> int:
        match = re.search(r'Sym\((-?\d+)\)', annotation)
        return int(match.group(1)) if match else 32

    def parse_dim(dim: str) -> int:
        try:
            return int(dim)
        except ValueError:
            return 32

    sig = fwd_match.group(1)
    inputs = []
    for arg in split_signature_args(sig):
        match = re.match(r'\w+\s*:\s*"([^"]*)"', arg)
        if match is None:
            inputs.append({"kind": "scalar", "value": 1})
            continue

        annotation = match.group(1)
        if annotation.startswith("Sym("):
            inputs.append({"kind": "symint", "hint": parse_symbol_hint(annotation)})
            continue

        m = re.match(
            r'(\w+)\[([^\]]*)\](?:\[([^\]]*)\])?(.*)',
            annotation,
        )
        if m is None:
            inputs.append({"kind": "scalar", "value": 1})
            continue

        dtype_str, shape_str, stride_str, device_str = m.groups()
        dims = []
        for x in shape_str.split(','):
            x = x.strip()
            if not x:
                continue
            dims.append(parse_dim(x))
        stride = None
        if stride_str is not None:
            stride = []
            for x in stride_str.split(','):
                x = x.strip()
                if not x:
                    continue
                try:
                    stride.append(parse_dim(x))
                except ValueError:
                    stride = None
                    break
            if stride is not None:
                stride = tuple(stride)
        shape = tuple(dims)
        dtype = dtype_map.get(dtype_str, torch.float32)
        inputs.append({
            "shape": shape,
            "dtype": dtype,
            "stride": stride,
            "device": device_str.strip() or None,
        })

    return inputs


def _forward_takes_no_inputs(content: str) -> bool:
    import re

    fwd_match = re.search(r'def forward\(self,?\s*(.*?)\):', content, re.DOTALL)
    return fwd_match is not None and not fwd_match.group(1).strip()


def _specs_have_symbols(specs) -> bool:
    """True if any parsed input spec carries a symbolic dim/stride/symint —
    i.e. the saved graph is dynamic and must be re-traced symbolically."""
    for s in specs:
        if not isinstance(s, dict):
            continue
        if s.get("kind") == "symint" and s.get("expr") is not None:
            return True
        for key in ("shape", "stride"):
            for d in (s.get(key) or []):
                if isinstance(d, str):
                    return True
        sym = s.get("symbolic") or {}
        if sym.get("shape_exprs") or sym.get("stride_exprs"):
            return True
    return False


def _sidecar_symbol_hints(graph_path: Path) -> dict:
    """Per-symbol NATIVE hints recorded at capture time in the graph's sidecar
    meta.json (full_graph_XXX.meta.json -> {"symbols": {"s16": {"hint": 4}}}).

    The saved .py annotations are symbolic (Sym(s16)) and carry no hint, so
    without the sidecar the re-trace seeds every symbol at a benign default.
    The default is fine for FIDELITY (the symbol stays symbolic either way)
    but not for the recorded POINT: the point's bindings become the default
    bench binding and its shape_hash (which includes hints) is the join key
    back to live-captured occurrence counts — a non-native hint times the
    wrong problem size and orphans the accounting join. Missing/corrupt
    sidecar -> {} (caller falls back to the default)."""
    meta_path = graph_path.with_name(graph_path.stem + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return {}
    hints = {}
    for name, defn in (meta.get("symbols") or {}).items():
        if not isinstance(defn, dict):
            continue
        hint = defn.get("hint")
        # bool is an int subclass but never a valid size hint (True would seed
        # a 1 -> ShapeEnv 0/1-specialization); a float that is exactly integral
        # is coerced losslessly (JSON may round-trip 4 as 4.0). Anything else
        # (None, non-integral float, str) is dropped -> caller's default hint.
        if isinstance(hint, bool):
            continue
        if isinstance(hint, int):
            hints[name] = hint
        elif isinstance(hint, float) and hint.is_integer():
            hints[name] = int(hint)
    return hints


def _build_symbolic_inputs(specs, dev, symbol_hints=None):
    """Rebuild the forward inputs for a dynamic saved graph with ONE shared
    ShapeEnv symbol per symbol NAME, so a symint input (arg2_1:Sym(s16)) and a
    tensor dynamic dim (arg4_1:[...,s16,...]) trace as the SAME symbol — the
    structure the model actually had. Returns (FakeTensorMode, inputs), or None
    if a symbol could not be resolved (caller falls back to real-mode trace).

    Symbols are created backed at their recorded hint (size_hint), mirroring
    capture; the strides come from the annotation's stride exprs evaluated
    over the same symbols, so a non-contiguous dynamic tensor round-trips.
    """
    import torch
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._dynamo.source import LocalSource

    # Collect symbol names + a hint per symbol. Precedence: the sidecar
    # meta.json's NATIVE hints (symbol_hints) first — they set the recaptured
    # point's bindings and shape_hash, which must match the live capture —
    # then a symint spec's own hint, then a benign default (8). The hint only
    # picks which concrete size make_fx traces at — the symbol stays symbolic
    # regardless — so the default is safe for structure, wrong for the point.
    hints: dict[str, int] = dict(symbol_hints or {})

    import ast as _ast

    def _expr_symbols(text: str) -> set:
        """Symbol identifiers referenced by a closed arithmetic expr, via AST
        (no regex, lossless over the printed grammar): 's0*s53' -> {'s0','s53'},
        's53' -> {'s53'}, '64' -> set(). Call-position names are NOT symbols —
        'CeilToInt(s0/3)' references only s0; seeding 'CeilToInt' would mint a
        bogus unused ShapeEnv symbol. A token the grammar doesn't cover
        (SyntaxError) contributes no symbols, so the caller later falls back to
        real-mode rather than seeding a half-parsed name."""
        try:
            tree = _ast.parse(text, mode="eval")
        except (SyntaxError, ValueError):
            return set()
        called = {n.func.id for n in _ast.walk(tree)
                  if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)}
        return {n.id for n in _ast.walk(tree)
                if isinstance(n, _ast.Name)} - called

    # A live symint carries an expr ('s53' or a product 's0*s53'); seed EVERY
    # symbol it references, not just bare-identifier exprs — a composite symint
    # whose components appear in no tensor dim must still resolve.
    for s in specs:
        if not isinstance(s, dict):
            continue
        if s.get("kind") == "symint" and isinstance(s.get("expr"), str):
            for nm in _expr_symbols(s["expr"]):
                hints.setdefault(nm, int(s.get("hint", 8) or 8))
    # Seed any still-unhinted symbols that appear in tensor SHAPE *or* STRIDE
    # tokens. Strides are seeded too: a symbol that occurs ONLY in a stride
    # expr (e.g. a permuted/non-contiguous dynamic tensor) would otherwise be
    # unknown to eval_dim, dropping the strided view to a contiguous fallback.
    for s in specs:
        if not isinstance(s, dict) or s.get("kind") == "symint":
            continue
        for tok in list(s.get("shape") or []) + list(s.get("stride") or []):
            if isinstance(tok, str):
                for nm in _expr_symbols(tok):
                    hints.setdefault(nm, 8)
    if not hints:
        return None

    shape_env = ShapeEnv()
    symnodes: dict[str, object] = {}
    trace_hints: dict[str, int] = {}
    for nm, h in hints.items():
        # ShapeEnv 0/1-specializes size hints of 0 and 1 — the created "symbol"
        # collapses to a constant, so a dynamic dim would silently recapture as
        # static. The hint only picks the concrete size make_fx traces at (the
        # symbol stays symbolic regardless), so trace at >=2 to keep it dynamic.
        trace_h = h if isinstance(h, int) and h >= 2 else 2
        trace_hints[nm] = trace_h
        sym = shape_env.create_symbol(
            trace_h, source=LocalSource(nm), dynamic_dim=DimDynamic.DYNAMIC)
        symnodes[nm] = shape_env.create_symintnode(sym, hint=trace_h)

    # Evaluate a dim/stride token to a torch SymInt (or int), covering the
    # SAME closed expression grammar input_codec accepts everywhere else —
    # not just +/-/* products but the torch shape functions a saved
    # annotation can legitimately carry ('s0//2', 'CeilToInt(s0/3)',
    # 'PythonMod(s0, 4)', 'ModularIndexing(...)'). The old evaluator's
    # charset gate rejected those, so a VALID symbolic dim returned None and
    # the whole graph silently recaptured STATIC — dynamism lost,
    # f(f(x)) != f(x) (PR80 review finding 4). _sympify_expr is the shared
    # safe-grammar boundary (AST allowlist gates the string before sympify
    # eval()s it; torch's sympy function classes parse natively); the parsed
    # expr is rebuilt over the ShapeEnv's OWN backing symbols and minted into
    # a torch SymInt via create_symintnode, so composite dims stay tied to
    # the same symbols as every other dim. An expr outside the grammar or
    # over unknown symbols returns None and the caller fails ingestion
    # loudly rather than guessing.
    from input_codec import _sympify_expr

    def eval_dim(tok):
        if isinstance(tok, int):
            return tok
        if not isinstance(tok, str):
            return tok
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            return int(tok)
        if tok.isidentifier():
            return symnodes.get(tok)  # bare symbol (None if unknown)
        try:
            expr = _sympify_expr(tok)
        except ValueError:
            return None  # outside the closed safe grammar — never guess
        free = list(getattr(expr, "free_symbols", ()))
        if any(s.name not in symnodes for s in free):
            return None
        # Concrete value at the trace hints — create_symintnode needs it, and
        # a non-integer fold means a function we can't evaluate: refuse.
        hint_val = expr.subs({s: trace_hints[s.name] for s in free})
        if not getattr(hint_val, "is_Integer", False):
            return None
        if not free:
            return int(hint_val)  # constant expr folds to a plain int
        sub = expr.subs({s: symnodes[s.name].node.expr for s in free})
        return shape_env.create_symintnode(sub, hint=int(hint_val))

    fake_mode = FakeTensorMode(shape_env=shape_env)

    def _is_contiguous(shape, strides):
        """Row-major contiguous check that works symbolically: compare each
        stride to the running product of trailing dims using == over SymInts
        (guarded). If any comparison can't be decided, treat as non-contiguous
        (build the explicit strided view) rather than guess."""
        running = 1
        for st, d in zip(reversed(strides), reversed(shape)):
            same = (st == running)
            if isinstance(same, bool):
                if not same:
                    return False
            else:
                return False  # symbolic-undecided -> use as_strided
            running = running * d
        return True

    def build(spec):
        kind = spec.get("kind")
        if kind == "symint":
            nm = spec.get("expr")
            if nm is None:
                # Constant-valued symint (Sym(256)): no symbol — keep the
                # recorded concrete value, don't collapse it to the default.
                return spec.get("value", spec.get("hint", 8))
            # Bare symbol ('s53') or composite ('s0*s53'): evaluate over the
            # shared symbol nodes so it stays a torch SymInt tied to the SAME
            # symbols as the tensor dims. An unknown component -> None -> caller
            # falls back to real-mode tracing (never silently bakes a guess).
            return eval_dim(nm)
        if kind == "scalar":
            return spec.get("value", 1)
        shape = [eval_dim(d) for d in (spec.get("shape") or [])]
        if any(d is None for d in shape):
            return None
        dtype = spec.get("dtype") or torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float32)
        device_text = spec.get("device") or ""
        tdev = torch.device("cuda:0") if device_text.startswith("cuda") and \
            torch.cuda.is_available() else dev
        stride_spec = spec.get("stride")
        with fake_mode:
            if stride_spec and len(stride_spec) == len(shape):
                strides = [eval_dim(st) for st in stride_spec]
                if not any(st is None for st in strides) and \
                        not _is_contiguous(shape, strides):
                    # storage big enough for the (symbolic) strided view; the
                    # arithmetic stays in torch SymInt space (shape/strides are
                    # SymInts), so torch.empty/as_strided accept it.
                    storage = 1
                    for st, d in zip(strides, shape):
                        storage = storage + st * (d - 1)
                    base = torch.empty([storage], dtype=dtype, device=tdev)
                    return torch.as_strided(base, shape, strides)
            return torch.empty(shape, dtype=dtype, device=tdev)

    inputs = []
    for spec in specs:
        built = build(spec)
        if built is None:
            return None  # unresolved -> let caller fall back to real mode
        inputs.append(built)
    return fake_mode, inputs


def load_graph_module(graph_path: Path):
    """Load a post-grad graph txt or fx_graph_runnable as an FX GraphModule.

    Handles two formats:
    - inductor_post_grad_graph_*.txt: has `class <lambda>(torch.nn.Module):`
    - fx_graph_runnable_*.txt: has `from torch.nn import *`, class Repro,
      __init__ with submodule constructors, and `if __name__` block.

    Strategy:
    1. Exec file content with globals that include torch.nn.* exports
    2. Strip __main__ blocks to avoid running the graph
    3. Replace `class <lambda>` with `class Repro`
    4. Trace the instantiated module with make_fx to produce a GraphModule
    """
    import re
    import torch
    import torch.nn
    import torch.fx as fx
    import torch._inductor.inductor_prims  # noqa: F401 - registers prims.fma, inductor seeds

    content = graph_path.read_text()

    # Fix 1: Replace `class <lambda>` with `class Repro` (invalid Python syntax)
    content = content.replace("class <lambda>", "class Repro")

    # Fix 2: Strip `if __name__ == '__main__':` blocks to avoid running the graph.
    # These blocks compile/run the model which we don't want.
    content = re.sub(
        r'^if __name__\s*==\s*[\'"]__main__[\'"]:\s*\n(?:(?:[ \t]+.*)?\n)*',
        '', content, flags=re.MULTILINE
    )

    # Also strip `mod = Repro()` lines at module level that appear before __main__
    # (the file creates the instance itself which we'll do ourselves)
    content = re.sub(r'^mod\s*=\s*Repro\(\)\s*$', '', content, flags=re.MULTILINE)

    # Extract input shapes BEFORE exec (from the original content)
    input_shapes = _parse_input_shapes_from_reader(content)
    if not input_shapes:
        input_shapes = _parse_input_shapes_from_annotations(content)

    # Build globals dict with all needed imports pre-populated.
    # This ensures `from torch.nn import *` style usage works (Module, etc.)
    exec_globals = {"__builtins__": __builtins__}

    # Add torch.nn.* to globals (handles AdaptiveAvgPool1d, Module, etc.)
    for attr in dir(torch.nn):
        if not attr.startswith('_'):
            exec_globals[attr] = getattr(torch.nn, attr)

    # Add core imports
    exec_globals['torch'] = torch
    exec_globals['fx'] = fx
    exec_globals['inf'] = math.inf
    exec_globals['nan'] = float('nan')
    exec_globals['device'] = torch.device
    exec_globals['tensor'] = torch.tensor

    try:
        exec(compile(content, str(graph_path), "exec"), exec_globals)

        # Find the class (named 'Repro' after our lambda fix, or other nn.Module subclass)
        graph_cls = None
        for name, v in exec_globals.items():
            if (isinstance(v, type)
                    and issubclass(v, torch.nn.Module)
                    and v is not torch.nn.Module
                    and name not in dir(torch.nn)):
                graph_cls = v
                break

        if graph_cls is None:
            return None

        # Instantiate -- this runs __init__ which may register buffers/submodules
        instance = graph_cls()

        # Recreate saved self.* tensor attributes referenced by printed full graphs.
        # Their values do not affect partitioning; shape/dtype/device do.
        dtype_short_map = {
            'f32': torch.float32, 'f16': torch.float16,
            'bf16': torch.bfloat16, 'i64': torch.int64,
            'i32': torch.int32, 'i16': torch.int16, 'b8': torch.bool,
            'f64': torch.float64, 'i8': torch.int8, 'u8': torch.uint8,
        }

        def parse_shape(shape_str: str) -> tuple[int, ...]:
            dims = []
            for dim in shape_str.split(','):
                dim = dim.strip()
                if not dim:
                    continue
                try:
                    dims.append(int(dim))
                except ValueError:
                    dims.append(1)
            return tuple(dims)

        def collect_self_tensor_attrs(prefix: str) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
            info = {}
            pattern = re.compile(
                rf'\w+:\s*"(\w+)\[([^\]]*)\][^"]*"\s*=\s*self\.({prefix}\w*)'
            )
            for line in content.splitlines():
                match = pattern.search(line)
                if not match:
                    continue
                dtype_str, shape_str, attr_name = match.groups()
                info[attr_name] = (
                    parse_shape(shape_str),
                    dtype_short_map.get(dtype_str, torch.float32),
                )
            return info

        tensor_attrs = {}
        # For inductor_post_grad_graph files: `_frozen_param*` buffers appear as:
        # `var: "dtype[shape][strides]device" = self._frozen_paramN`
        tensor_attrs.update(collect_self_tensor_attrs("_frozen_param"))
        # Saved full_graph_*.py files can also reference scalar/tensor constants:
        # `_tensor_constant0: "f32[]" = self._tensor_constant0`
        tensor_attrs.update(collect_self_tensor_attrs("_tensor_constant"))

        if tensor_attrs:
            dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            for attr_name, (shape, dtype) in tensor_attrs.items():
                if hasattr(instance, attr_name):
                    continue
                buf = torch.zeros(shape, dtype=dtype, device=dev)
                instance.register_buffer(attr_name, buf)

        # If already a GraphModule (has .graph), return directly
        if hasattr(instance, 'graph'):
            return instance

        # Otherwise, trace with make_fx to produce a proper GraphModule
        if not input_shapes and not _forward_takes_no_inputs(content):
            print(f"  No input shapes found for {graph_path.name}")
            return None

        from torch.fx.experimental.proxy_tensor import make_fx

        # Create inputs on CUDA (forward methods typically hardcode cuda:0 device)
        dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        instance = instance.to(dev)

        def normalize_input_spec(spec):
            if isinstance(spec, dict):
                return spec
            shape, dtype = spec
            return {"shape": shape, "dtype": dtype, "stride": None, "device": None}

        def make_tensor_from_spec(spec):
            if spec.get("kind") == "symint":
                return int(spec.get("hint", 1))
            if spec.get("kind") == "scalar":
                return spec.get("value", 1)

            shape = tuple(spec["shape"])
            dtype = spec["dtype"]
            stride = spec.get("stride")
            device_text = spec.get("device") or ""
            if device_text.startswith("cuda") and torch.cuda.is_available():
                tensor_dev = torch.device("cuda:0")
            elif device_text.startswith("cpu"):
                tensor_dev = torch.device("cpu")
            else:
                tensor_dev = dev

            if dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
                make_base = lambda size: torch.zeros(size, dtype=dtype, device=tensor_dev)
            elif dtype == torch.bool:
                make_base = lambda size: torch.zeros(size, dtype=torch.bool, device=tensor_dev)
            else:
                make_base = lambda size: torch.randn(size, dtype=dtype, device=tensor_dev)

            if stride is not None and len(stride) == len(shape) and shape:
                storage_size = sum(
                    s * (d - 1) for s, d in zip(stride, shape) if d > 1
                ) + 1
                return make_base((storage_size,)).as_strided(shape, stride)
            return make_base(shape)

        # FAITHFUL DYNAMIC RECAPTURE (idempotence): a saved full graph whose
        # forward annotations carry symbolic dims/symints (e.g. arg4_1:
        # "f32[64,64,s16,s82][64*s16*s82,...]" + arg2_1:"Sym(s16)") must be
        # re-traced SYMBOLICALLY, with the symint inputs sharing the SAME
        # symbols as the tensor dynamic dims — else make_fx(real) bakes them to
        # their hints and a dynamic graph recaptures as a STATIC region
        # (frozen dims + lifted _shape_param const list). The lossy local
        # annotation parser collapses s16->hint and strides->garbage, so use
        # full_graph_harness.parse_full_graph_inputs (keeps symbol names +
        # exprs) and reconstruct shared ShapeEnv symbols. Static graphs (no
        # symbolic spec) keep the original real-mode path untouched.
        from full_graph_harness import parse_full_graph_inputs
        dyn_specs = parse_full_graph_inputs(content)
        if dyn_specs and _specs_have_symbols(dyn_specs):
            sym_inputs = _build_symbolic_inputs(
                dyn_specs, dev, _sidecar_symbol_hints(graph_path))
            if sym_inputs is not None:
                fake_mode, inputs = sym_inputs
                with fake_mode:
                    gm = make_fx(instance, tracing_mode="symbolic")(*inputs)
                return gm
            # A dynamic graph whose symbolic rebuild failed (an unresolved
            # symbol, or an expr outside the closed grammar) must NOT
            # recapture as static: a static recapture of a dynamic family
            # silently corrupts point/family identity and breaks
            # f(f(x)) == f(x). FAIL the ingestion instead — a skipped graph
            # is visible and recoverable, a wrong recapture is neither
            # (PR80 review finding 4; supersedes the earlier warn-and-
            # -degrade behavior).
            print(f"  ERROR: {graph_path.name} carries symbolic inputs but "
                  f"symbolic rebuild failed (unresolved symbol/expr) — "
                  f"refusing to recapture a dynamic graph as static")
            return None

        inputs = []
        for raw_spec in input_shapes:
            spec = normalize_input_spec(raw_spec)
            inputs.append(make_tensor_from_spec(spec))

        gm = make_fx(instance, tracing_mode="real")(*inputs)
        return gm

    except Exception as e:
        print(f"  Failed to load {graph_path.name}: {e}")
        return None


def process_graph_with_hook(gm, label: str, output_dir: Path, suite: str):
    """Run the capture hook's partitioning on a loaded GraphModule."""
    import torch
    from capture_hook import _CaptureState
    from merge_captures import merge_one_capture

    cap_dir = Path(tempfile.mkdtemp())
    state = _CaptureState(str(cap_dir), label=label)

    try:
        state.process_graph(gm)
        state.finalize()

        index_path = cap_dir / "index.json"
        if not index_path.exists():
            return 0

        n = merge_one_capture(cap_dir, output_dir, label, suite=suite, mode="infer")
        return n
    except Exception as e:
        print(f"  Process failed for {label}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=None,
                        help="GitHub Actions run ID to download from")
    parser.add_argument("--tlparse-dir", type=Path, default=None,
                        help="Already-downloaded tlparse directory")
    parser.add_argument("--suite", type=str, default="torchbench",
                        choices=["torchbench", "huggingface", "timm", "all"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/tmp/scratch_space/better_benchmark/repros"))
    parser.add_argument("--list-only", action="store_true",
                        help="Just list available graphs, don't process")
    parser.add_argument("--use-runnables", action="store_true",
                        help="Use fx_graph_runnable files directly instead of post_grad_graph")
    args = parser.parse_args()

    if args.tlparse_dir:
        tlparse_dir = args.tlparse_dir
    elif args.run_id:
        tlparse_dir = Path(tempfile.mkdtemp()) / "tlparse"
        suite_filter = args.suite if args.suite != "all" else None
        download_tlparse(args.run_id, suite_filter, tlparse_dir)
    else:
        parser.error("Must provide --run-id or --tlparse-dir")

    if args.use_runnables:
        graphs = find_fx_graph_runnables(tlparse_dir)
        print(f"Found {len(graphs)} fx_graph_runnable files")
    else:
        graphs = find_post_grad_graphs(tlparse_dir)
        print(f"Found {len(graphs)} inductor_post_grad_graph files")

    if args.list_only:
        for g in graphs:
            print(f"  {g.relative_to(tlparse_dir)}")
        return

    import torch
    total_regions = 0
    for i, graph_path in enumerate(graphs):
        label = infer_model_name(graph_path) + f"_graph{i}"
        print(f"[{i+1}/{len(graphs)}] {graph_path.name} ({label})...", end=" ", flush=True)

        gm = load_graph_module(graph_path)
        if gm is None:
            print("SKIP (load failed)")
            continue

        import copy
        n = process_graph_with_hook(copy.deepcopy(gm), label, args.output_dir, args.suite)
        total_regions += n
        print(f"{n} regions")

    print(f"\nDone: {total_regions} regions from {len(graphs)} graphs")


if __name__ == "__main__":
    main()
