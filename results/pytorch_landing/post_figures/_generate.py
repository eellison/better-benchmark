#!/usr/bin/env python3
"""Generate three self-contained HTML figures for the perf-engineering blog post.

Data sources (all real, on disk):
  fig1: results/pytorch_landing/A1_commit_walk/FULL_per_commit_table.txt (+ _e2e.txt)
  fig2: perf_ab_rollup.py output staged at _data_permodel.json
  fig3: /tmp/scratch_space/perfetto_oracle_overlay_artifacts/PROOF_priced_overlay.perfetto.json

Palette (validated with dataviz skill's validate_palette.js, light mode, surface #fcfcfb):
  diverging pair  : blue #2a78d6 (positive) / red #e34948 (negative), neutral gray #f0efec  -> ALL PASS
  categorical 1,2 : blue #2a78d6 / aqua #1baf7a  -> PASS, aqua contrast WARN 2.74:1 (relief:
                    direct lane labels + lane totals + table view are shipped)
Ink/chrome from references/palette.md (light column).

No headless browser available on this host -> geometry is asserted here instead
(every mark/label extent checked against the SVG viewport and plot boxes).
"""
import json, re, html, statistics, math, os

OUT = "/tmp/scratch_space/better_benchmark/results/pytorch_landing/post_figures"

# ---------------- palette / chrome ----------------
POS   = "#2a78d6"   # diverging blue (positive / improvement)
NEG   = "#e34948"   # diverging red  (negative / regression)
CAT1  = "#2a78d6"   # categorical slot 1 (compile lane)
CAT2  = "#1baf7a"   # categorical slot 2 (oracle lane)
NEUT  = "#f0efec"   # neutral diverging midpoint gray (noise bands)
SURF  = "#fcfcfb"
PAGE  = "#f9f9f7"
INK   = "#0b0b0b"
INK2  = "#52514e"
MUTED = "#898781"
GRID  = "#e1e0d9"
AXIS  = "#c3c2b7"
FONT  = 'system-ui,-apple-system,"Segoe UI",sans-serif'

ASSERTS = []
def check(cond, msg):
    if not cond:
        raise AssertionError("GEOMETRY: " + msg)
    ASSERTS.append(msg)

def est_w(text, size):
    """conservative text-width estimate for sans (px)."""
    return len(text) * size * 0.62

def esc(s): return html.escape(str(s), quote=True)

def rbar_h(xb, xt, y, h, r=4):
    """horizontal bar: square at baseline xb, rounded at data end xt."""
    r = min(r, h / 2, abs(xt - xb))
    if xt >= xb:
        return (f"M{xb:.2f},{y:.2f} L{xt - r:.2f},{y:.2f} Q{xt:.2f},{y:.2f} {xt:.2f},{y + r:.2f} "
                f"L{xt:.2f},{y + h - r:.2f} Q{xt:.2f},{y + h:.2f} {xt - r:.2f},{y + h:.2f} "
                f"L{xb:.2f},{y + h:.2f} Z")
    return (f"M{xb:.2f},{y:.2f} L{xt + r:.2f},{y:.2f} Q{xt:.2f},{y:.2f} {xt:.2f},{y + r:.2f} "
            f"L{xt:.2f},{y + h - r:.2f} Q{xt:.2f},{y + h:.2f} {xt + r:.2f},{y + h:.2f} "
            f"L{xb:.2f},{y + h:.2f} Z")

def rbar_v(x, w, yb, yt, r=2.5):
    """vertical bar: square at baseline yb, rounded at data end yt."""
    r = min(r, w / 2, abs(yt - yb))
    if yt <= yb:  # grows upward
        return (f"M{x:.2f},{yb:.2f} L{x:.2f},{yt + r:.2f} Q{x:.2f},{yt:.2f} {x + r:.2f},{yt:.2f} "
                f"L{x + w - r:.2f},{yt:.2f} Q{x + w:.2f},{yt:.2f} {x + w:.2f},{yt + r:.2f} "
                f"L{x + w:.2f},{yb:.2f} Z")
    return (f"M{x:.2f},{yb:.2f} L{x:.2f},{yt - r:.2f} Q{x:.2f},{yt:.2f} {x + r:.2f},{yt:.2f} "
            f"L{x + w - r:.2f},{yt:.2f} Q{x + w:.2f},{yt:.2f} {x + w:.2f},{yt - r:.2f} "
            f"L{x + w:.2f},{yb:.2f} Z")

TOOLTIP_JS = """
(function(){
  var tip = document.getElementById('tip');
  var root = document.querySelector('.viz-root');
  function show(e, lines){
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    lines.forEach(function(ln, i){
      var d = document.createElement('div');
      if (i === 0) d.className = 'tip-val';
      d.textContent = ln;                       // untrusted data -> textContent
      tip.appendChild(d);
    });
    tip.style.display = 'block';
    move(e);
  }
  function move(e){
    var pad = 14, r = root.getBoundingClientRect();
    var x = e.clientX - r.left + pad, y = e.clientY - r.top + pad;
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    if (e.clientX + pad + tw > window.innerWidth - 8) x = e.clientX - r.left - pad - tw;
    if (e.clientY + pad + th > window.innerHeight - 8) y = e.clientY - r.top - pad - th;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hide(){ tip.style.display = 'none'; }
  document.querySelectorAll('[data-tip]').forEach(function(el){
    var lines = JSON.parse(el.getAttribute('data-tip'));
    function on(e){ el.classList.add('hov'); show(e, lines); }
    el.addEventListener('pointerenter', on);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerleave', function(){ el.classList.remove('hov'); hide(); });
    el.addEventListener('focus', function(){ var r = el.getBoundingClientRect();
      show({clientX:r.left + r.width/2, clientY:r.top}, lines); el.classList.add('hov'); });
    el.addEventListener('blur', function(){ el.classList.remove('hov'); hide(); });
    el.setAttribute('tabindex','0');
  });
})();
"""

def page(title, subtitle, svg, legend_html, foot, table_html, width):
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
  :root {{ color-scheme: light; }}
  html,body {{ margin:0; padding:0; background:{PAGE}; font-family:{FONT}; }}
  .viz-root {{ position:relative; width:{width}px; margin:28px auto; background:{SURF};
    border:1px solid rgba(11,11,11,0.10); border-radius:8px; padding:22px 26px 18px; box-sizing:border-box; }}
  h1 {{ font-size:17px; font-weight:650; color:{INK}; margin:0 0 3px; letter-spacing:-0.01em; }}
  .sub {{ font-size:13px; color:{INK2}; margin:0 0 14px; }}
  .legend {{ display:flex; gap:18px; align-items:center; font-size:12px; color:{INK2}; margin:2px 0 10px; }}
  .legend .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
  .foot {{ font-size:11px; color:{MUTED}; margin-top:10px; line-height:1.45; }}
  svg {{ display:block; }}
  svg text {{ font-family:{FONT}; }}
  #tip {{ display:none; position:absolute; z-index:10; pointer-events:none; background:#ffffff;
    border:1px solid rgba(11,11,11,0.14); border-radius:6px; box-shadow:0 4px 14px rgba(11,11,11,0.12);
    padding:7px 10px; font-size:12px; color:{INK2}; max-width:340px; }}
  #tip .tip-val {{ color:{INK}; font-weight:650; font-size:12.5px; margin-bottom:2px; }}
  .hov {{ opacity:0.82; }}
  rect.hit {{ fill:transparent; }}
  rect.hit.hov {{ fill:rgba(11,11,11,0.06); opacity:1; }}
  [data-tip] {{ cursor:default; outline:none; }}
  [data-tip]:focus-visible {{ outline:2px solid {INK2}; outline-offset:1px; }}
  details {{ margin-top:12px; font-size:12px; color:{INK2}; }}
  details summary {{ cursor:pointer; color:{MUTED}; }}
  table {{ border-collapse:collapse; margin-top:8px; font-size:11.5px; }}
  th,td {{ text-align:left; padding:2.5px 12px 2.5px 0; border-bottom:1px solid {GRID}; }}
  td.num,th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
</style></head>
<body>
<figure class="viz-root" style="margin:28px auto;">
  <h1>{esc(title)}</h1>
  <p class="sub">{esc(subtitle)}</p>
  {legend_html}
  {svg}
  <p class="foot">{foot}</p>
  {table_html}
  <div id="tip" role="status"></div>
</figure>
<script>{TOOLTIP_JS}</script>
</body></html>
"""

# =====================================================================
# FIGURE 1 — per-commit attribution, two aligned diverging panels
# =====================================================================
def parse_commit_table(path):
    rows = []
    for line in open(path):
        m = re.match(r"\s*pos\s*(\d+)->\s*(\d+)\s+([0-9a-f]+)\s+([+-]\d+\.\d+)pp\s+(.*)$", line)
        if m:
            rows.append({"sha": m.group(3), "pp": float(m.group(4)), "subj": m.group(5).strip()})
    return rows

SHORT = {
    "97385fb3273": "kernel-opt mega-commit",
    "ab29345d82d": "CE loop-invariant hoist",
    "f9493caabcd": "pointwise_cat guard relax",
    "ca8f961d6b0": "scalar_acc load gate",
    "94ad050dee0": "select_scatter sparsity pass",
    "1a824e0747a": "realize_reads threshold 4→30",
    "f74a7f13723": "revert: realize_reads threshold",
    "afc3d5956d5": "bounds-check elision",
    "905450a5a5d": "inline recomputable (scheduler)",
    "f595e0cac3a": "select_scatter topo-order fix",
    "a73d1583b34": "rsqrt canonicalization",
    "86c5451e197": "BN affine fold (net-neg)",
    "f0b54e1aa1f": "revert: num_stages tuning",
    "d2cd52d3117": "persistent-reduction threshold",
    "62f27911a59": "revert: persistent-red. threshold",
    "b1e8a30af15": "materialize transcendentals",
    "4427bfd553c": "scatter_add fusion dtypes",
    "d8e9914094a": "scatter read bypass (Longformer)",
    "d95f59fb1bc": "pointwise_cat shared-reads fix",
    "44802547aa9": "nd-reduction assert-crash fix",
    "1406552b9d3": "RoPE rotate-half gather",
    "5489b8c2bb9": "scatter_add→reduction elim",
    "bdc289b3644": "multi-output reduction split",
    "e9a67c98a8c": "evict_first restriction",
    "a26fc2c8bf4": "online-softmax fast combine",
    "60dd3839913": "dedupe outputs (earliest node)",
    "56959375c33": "affine-iota index decode",
    "a85d79a900a": "segment-aligned split",
    "9dde2c59a51": "scalar-acc ungate + rnumel",
    "daa79cd25ca": "dedupe outputs (structural hash)",
}
LABELED = {"a73d1583b34", "97385fb3273", "ab29345d82d", "86c5451e197"}  # direct-label only these

def fig1():
    base = "/tmp/scratch_space/better_benchmark/results/pytorch_landing/A1_commit_walk"
    kern = parse_commit_table(base + "/FULL_per_commit_table.txt")
    e2e  = parse_commit_table(base + "/FULL_per_commit_e2e.txt")
    e2e_by = {r["sha"]: r for r in e2e}
    rows = []
    for r in kern:
        if r["subj"].startswith("Update"):
            continue  # merge-noise rows excluded per spec
        e = e2e_by[r["sha"]]
        rows.append({"sha": r["sha"], "k": r["pp"], "e": e["pp"],
                     "subj": r["subj"], "name": SHORT[r["sha"]]})
    n = len(rows)
    check(n == 30, f"fig1 has {n} named commits (expected 30)")

    # ---- geometry ----
    W = 1060
    pitch, barh = 25, 16
    top, bot = 40, 40
    H = top + n * pitch + bot
    lab_r = 250                       # right edge of row labels
    pAx, pAw = 260, 358
    pBx, pBw = pAx + pAw + 28, 358
    check(pBx + pBw <= W - 14, "fig1 panel B inside viewport")

    kdom = (-1.3, 6.4)
    edom = (-1.05, 2.55)
    def xk(v): return pAx + (v - kdom[0]) / (kdom[1] - kdom[0]) * pAw
    def xe(v): return pBx + (v - edom[0]) / (edom[1] - edom[0]) * pBw
    kmin, kmax = min(r["k"] for r in rows), max(r["k"] for r in rows)
    emin, emax = min(r["e"] for r in rows), max(r["e"] for r in rows)
    check(kdom[0] < kmin and kmax < kdom[1], f"fig1 kernel domain covers data [{kmin},{kmax}]")
    check(edom[0] < emin and emax < edom[1], f"fig1 e2e domain covers data [{emin},{emax}]")

    s = []
    s.append(f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Per-commit marginal delta, kernel geomean and per-model e2e">')
    # panel titles
    s.append(f'<text x="{pAx}" y="20" font-size="12.5" font-weight="650" fill="{INK}">Kernel-geomean Δpp</text>')
    s.append(f'<text x="{pBx}" y="20" font-size="12.5" font-weight="650" fill="{INK}">Per-model e2e Δpp</text>')

    plot_y0, plot_y1 = top, top + n * pitch
    # noise bands (neutral gray)
    for x0, x1, px, pw in ((xk(-0.2), xk(0.2), pAx, pAw), (xe(-0.82), xe(0.82), pBx, pBw)):
        s.append(f'<rect x="{x0:.2f}" y="{plot_y0}" width="{x1 - x0:.2f}" height="{n * pitch}" fill="{NEUT}"/>')
    # band notes (below plot, above axis labels)
    s.append(f'<text x="{xk(0):.2f}" y="{plot_y1 + 30}" font-size="10" fill="{MUTED}" text-anchor="middle">'
             f'gray band = ±0.2pp noise floor</text>')
    s.append(f'<text x="{xe(0):.2f}" y="{plot_y1 + 30}" font-size="10" fill="{MUTED}" text-anchor="middle">'
             f'gray band = ±0.82pp noise floor</text>')

    # gridlines + ticks
    for v in (0, 2, 4, 6):
        x = xk(v)
        s.append(f'<line x1="{x:.2f}" y1="{plot_y0}" x2="{x:.2f}" y2="{plot_y1}" stroke="{GRID if v else AXIS}" stroke-width="1"/>')
        s.append(f'<text x="{x:.2f}" y="{plot_y1 + 15}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{"%+d" % v if v else "0"}</text>')
    for v in (-1, 0, 1, 2):
        x = xe(v)
        s.append(f'<line x1="{x:.2f}" y1="{plot_y0}" x2="{x:.2f}" y2="{plot_y1}" stroke="{GRID if v else AXIS}" stroke-width="1"/>')
        s.append(f'<text x="{x:.2f}" y="{plot_y1 + 15}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{"%+d" % v if v else "0"}</text>')

    # rows
    for i, r in enumerate(rows):
        y = top + i * pitch
        yb = y + (pitch - barh) / 2
        name = r["name"]
        check(est_w(name, 11.5) <= lab_r - 8, f"fig1 row label fits: {name}")
        s.append(f'<text x="{lab_r}" y="{y + pitch / 2 + 4:.2f}" font-size="11.5" fill="{INK2}" text-anchor="end">{esc(name)}</text>')
        tipk = json.dumps([f"kernel {r['k']:+.2f}pp · e2e {r['e']:+.2f}pp", r["subj"], r["sha"]])
        for val, xf, x0f in ((r["k"], xk, xk(0)), (r["e"], xe, xe(0))):
            xt = xf(val)
            col = POS if val >= 0 else NEG
            s.append(f'<path d="{rbar_h(x0f, xt, yb, barh)}" fill="{col}"/>')
        # row-wide hit area (bars as thin as 1px are pinpoint targets otherwise)
        s.append(f'<rect class="hit" x="0" y="{y:.2f}" width="{pBx + pBw:.2f}" height="{pitch}" data-tip="{esc(tipk)}"/>')
        if r["sha"] in LABELED:
            for val, xf in ((r["k"], xk), (r["e"], xe)):
                xt = xf(val)
                lab = f"{val:+.2f}"
                if val >= 0:
                    lx, anc = xt + 5, "start"
                else:
                    # negative bar: tip-side would collide with the row label;
                    # the space right of the zero line is empty on this row
                    lx, anc = xf(0) + 5, "start"
                check(lx + est_w(lab, 11) <= (pAx + pAw if xf is xk else pBx + pBw) + 24,
                      f"fig1 tip label in bounds: {lab}")
                s.append(f'<text x="{lx:.2f}" y="{yb + barh - 3.5:.2f}" font-size="11" font-weight="650" fill="{INK}" text-anchor="{anc}">{esc(lab)}</text>')

    # axis unit captions
    s.append(f'<text x="{pAx + pAw}" y="{plot_y1 + 15}" font-size="10.5" fill="{MUTED}" text-anchor="end">pp</text>')
    s.append(f'<text x="{pBx + pBw}" y="{plot_y1 + 15}" font-size="10.5" fill="{MUTED}" text-anchor="end">pp</text>')
    s.append('</svg>')

    legend = (f'<div class="legend"><span><span class="sw" style="background:{POS}"></span>improvement (+pp)</span>'
              f'<span><span class="sw" style="background:{NEG}"></span>regression (−pp)</span>'
              f'<span style="color:{MUTED}">commits in branch order, top → bottom · same rows, two measures</span></div>')
    foot = ("Marginal delta of each named commit vs the previous benched commit (5e2ab… → daa79…), genai-excluded. "
            "“Update” merge rows excluded as noise. Kernel-geomean overstates model impact ~2–3×; "
            "the right panel is the honest per-model end-to-end number (fusible+extern denominator).")
    trows = "".join(
        f"<tr><td>{esc(r['name'])}</td><td><code>{esc(r['sha'])}</code></td>"
        f"<td class='num'>{r['k']:+.2f}</td><td class='num'>{r['e']:+.2f}</td></tr>" for r in rows)
    table = ("<details><summary>Data table (30 commits)</summary><table><tr><th>commit</th><th>sha</th>"
             "<th class='num'>kernel Δpp</th><th class='num'>e2e Δpp</th></tr>" + trows + "</table></details>")
    return page("Which commits carry the win — per-commit marginal delta",
                "kernel-geomean vs honest per-model e2e (extern-diluted ~2–3×)",
                "".join(s), legend, foot, table, W + 54)

# =====================================================================
# FIGURE 2 — 158-model sorted bar strip
# =====================================================================
def fig2():
    d = json.load(open(OUT + "/_data_permodel.json"))
    pm = {k: v for k, v in d["per_model"].items() if v is not None and "genai" not in k}
    models = sorted(pm.items(), key=lambda kv: -kv[1])
    n = len(models)
    check(n == 158, f"fig2 has {n} models (expected 158)")
    med = statistics.median(v for _, v in models)

    W = 1020
    ml, mr, mt, mb = 52, 16, 66, 34
    pw, ph = W - ml - mr, 400
    H = mt + ph + mb
    pitch = pw / n
    barw = pitch - 1.0                     # 1px gap at this density (158 bars)
    check(barw >= 4.0, f"fig2 bar width {barw:.2f}px >= 4px")
    ydom = (-23.5, 29.5)
    def Y(v): return mt + (ydom[1] - v) / (ydom[1] - ydom[0]) * ph
    vmin, vmax = models[-1][1], models[0][1]
    check(ydom[0] < vmin and vmax < ydom[1], f"fig2 domain covers [{vmin:.1f},{vmax:.1f}]")

    s = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Per-model end-to-end change for 158 models, sorted">']
    # gridlines + y ticks
    for v in (-20, -10, 10, 20):
        y = Y(v)
        s.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{ml + pw}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{ml - 8}" y="{y + 3.5:.2f}" font-size="10.5" fill="{MUTED}" text-anchor="end">{v:+d}%</text>')
    # noise band
    yb0, yb1 = Y(0.82), Y(-0.82)
    s.append(f'<rect x="{ml}" y="{yb0:.2f}" width="{pw}" height="{yb1 - yb0:.2f}" fill="{NEUT}"/>')
    s.append(f'<text x="{ml + pw}" y="{yb1 + 13:.2f}" font-size="10" fill="{MUTED}" text-anchor="end">'
             f'±0.82% A/A noise — within measurement noise</text>')
    # zero baseline
    s.append(f'<line x1="{ml}" y1="{Y(0):.2f}" x2="{ml + pw}" y2="{Y(0):.2f}" stroke="{AXIS}" stroke-width="1"/>')
    s.append(f'<text x="{ml - 8}" y="{Y(0) + 3.5:.2f}" font-size="10.5" fill="{MUTED}" text-anchor="end">0</text>')

    # bars + full-height column hit areas (bars ~5px wide are pinpoint targets otherwise)
    hits2 = []
    for i, (k, v) in enumerate(models):
        x = ml + i * pitch
        col = POS if v >= 0 else NEG
        tip = json.dumps([f"{v:+.2f}% e2e", k, f"rank {i + 1} of {n}"])
        s.append(f'<path d="{rbar_v(x, barw, Y(0), Y(v))}" fill="{col}"/>')
        hits2.append(f'<rect class="hit" x="{x:.2f}" y="{mt}" width="{pitch:.2f}" height="{ph}" data-tip="{esc(tip)}"/>')
        check(ml <= x and x + barw <= ml + pw + 0.01, f"fig2 bar {i} within plot")

    # median line + label
    ym = Y(med)
    s.append(f'<line x1="{ml}" y1="{ym:.2f}" x2="{ml + pw}" y2="{ym:.2f}" stroke="{INK2}" stroke-width="1"/>')
    s.append(f'<text x="{ml + pw - 4}" y="{ym - 5:.2f}" font-size="11" font-weight="650" fill="{INK}" '
             f'text-anchor="end">median {med:+.2f}%</text>')

    # direct labels: top 3 (leftmost bars) via leader lines
    def leader(i, v, ty, anchor_left):
        bx = ml + i * pitch + barw / 2
        by = Y(v) - 2
        tx = ml + 26 + (0 if anchor_left else 0)
        return bx, by, tx, ty
    top3 = models[:3]
    for j, (k, v) in enumerate(top3):
        name = k.split("/")[-1]
        lab = f"{name} {v:+.1f}%"
        bx = ml + j * pitch + barw / 2
        by = Y(v) - 3
        ty = mt - 46 + j * 15
        tx = ml + 34
        s.append(f'<path d="M{bx:.2f},{by:.2f} L{bx:.2f},{ty + 9:.2f} L{tx - 4:.2f},{ty + 9:.2f}" '
                 f'fill="none" stroke="{MUTED}" stroke-width="1"/>')
        s.append(f'<text x="{tx}" y="{ty + 12:.2f}" font-size="11" fill="{INK}" font-weight="600">{esc(lab)}</text>')
        check(ty + 12 > 10 and tx + est_w(lab, 11) < W, f"fig2 top label fits: {lab}")
    # regressions (rightmost region): label the 3 named ones
    regs = [("torchbench/infer/pytorch_unet",), ("torchbench/infer/pyhpc_turbulent_kinetic_energy",),
            ("hf/infer/AllenaiLongformerBase",)]
    idx = {k: (i, v) for i, (k, v) in enumerate(models)}
    reg_disp = {"torchbench/infer/pytorch_unet": "pytorch_unet",
                "torchbench/infer/pyhpc_turbulent_kinetic_energy": "pyhpc_turb_kinetic_energy",
                "hf/infer/AllenaiLongformerBase": "AllenaiLongformerBase"}
    for j, (key,) in enumerate(regs):
        i, v = idx[key]
        lab = f"{reg_disp[key]} {v:+.1f}%"
        bx = ml + i * pitch + barw / 2
        by = Y(v) + 3
        ty = mt + ph - 44 + j * 15
        tw = est_w(lab, 11)
        tx = ml + pw - tw - 70
        s.append(f'<path d="M{bx:.2f},{by:.2f} L{bx:.2f},{ty - 4:.2f} L{tx + tw + 4:.2f},{ty - 4:.2f}" '
                 f'fill="none" stroke="{MUTED}" stroke-width="1"/>')
        s.append(f'<text x="{tx}" y="{ty:.2f}" font-size="11" fill="{INK}" font-weight="600" text-anchor="start">{esc(lab)}</text>')
        check(tx > ml and tx + tw < ml + pw and ty < H - mb + 10, f"fig2 reg label fits: {lab}")
    # x caption
    s.append(f'<text x="{ml + pw / 2:.2f}" y="{H - 8}" font-size="10.5" fill="{MUTED}" text-anchor="middle">'
             f'158 models, sorted by e2e change (best → worst)</text>')
    s.extend(hits2)  # hit layer on top
    s.append('</svg>')

    legend = (f'<div class="legend"><span><span class="sw" style="background:{POS}"></span>faster after</span>'
              f'<span><span class="sw" style="background:{NEG}"></span>slower after</span>'
              f'<span style="color:{MUTED}">y = per-model end-to-end % change, base 5e2ab… → head daa79…</span></div>')
    foot = ("Per-model e2e change over the full landing stack (genai-excluded, 158 models; "
            "fusible+extern denominator via perf_ab_rollup.py). Hover any bar for the model name and value; "
            "full listing in the data table below.")
    trows = "".join(f"<tr><td>{esc(k)}</td><td class='num'>{v:+.2f}%</td></tr>" for k, v in models)
    table = ("<details><summary>Data table (158 models)</summary><table>"
             "<tr><th>model</th><th class='num'>e2e change</th></tr>" + trows + "</table></details>")
    return page("158 models, before → after: a concentrated win",
                "per-model end-to-end change; most models within noise, a few conv/BN models carry the geomean",
                "".join(s), legend, foot, table, W + 54), med, models

# =====================================================================
# FIGURE 3 — execution-ordered timeline, compile vs oracle ceiling
# =====================================================================
def fig3():
    d = json.load(open("/tmp/scratch_space/perfetto_oracle_overlay_artifacts/PROOF_priced_overlay.perfetto.json"))
    ev = d["traceEvents"]
    comp = sorted([e for e in ev if e.get("ph") == "X" and e["tid"] == 1], key=lambda e: e["ts"])
    orc  = sorted([e for e in ev if e.get("ph") == "X" and e["tid"] == 3], key=lambda e: e["ts"])
    check(len(comp) == 46 and len(orc) == 46, "fig3 has 46 kernel pairs")
    pairs = list(zip(comp, orc))
    for c, o in pairs:
        check(abs(c["ts"] - o["ts"]) < 1e-6, "fig3 pair shares start ts")
    total_c = sum(c["dur"] for c in comp)
    total_o = sum(o["dur"] for o in orc)
    tmax = max(c["ts"] + c["dur"] for c in comp)

    W = 1080
    ml, mr = 24, 128
    pw = W - ml - mr
    lane_h = 24
    yl1, yl2 = 58, 122        # lane tops
    ax_y = yl2 + lane_h + 26  # time axis baseline
    H = ax_y + 26
    def X(us): return ml + us / tmax * pw

    s = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Execution-ordered kernel timeline, current compile vs oracle ceiling">']
    # lane labels (above each lane)
    s.append(f'<text x="{ml}" y="{yl1 - 8}" font-size="12" font-weight="650" fill="{INK}">torch.compile (current) '
             f'<tspan fill="{MUTED}" font-weight="400">— 46 fused kernels, laid end to end</tspan></text>')
    s.append(f'<text x="{ml}" y="{yl2 - 8}" font-size="12" font-weight="650" fill="{INK}">reference-kernel ceiling (oracle) '
             f'<tspan fill="{MUTED}" font-weight="400">— same kernels, same start; whitespace after each bar = headroom</tspan></text>')

    # time axis + gridlines
    for us in range(0, 15000, 2000):
        x = X(us)
        s.append(f'<line x1="{x:.2f}" y1="{yl1 - 2}" x2="{x:.2f}" y2="{ax_y}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x:.2f}" y="{ax_y + 16}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{us:,}</text>')
    s.append(f'<line x1="{ml}" y1="{ax_y}" x2="{ml + pw}" y2="{ax_y}" stroke="{AXIS}" stroke-width="1"/>')
    s.append(f'<text x="{ml + pw}" y="{ax_y + 16}" font-size="10.5" fill="{MUTED}" text-anchor="start">µs</text>')

    # slices; compile slices are contiguous -> shave 0.8px for a surface gap (min width 0.6px)
    GAP = 0.8
    hit_rects = []
    biggest = max(pairs, key=lambda p: p[0]["dur"] - p[1]["dur"])
    for i, (c, o) in enumerate(pairs):
        x0 = X(c["ts"])
        wc = max(0.6, X(c["ts"] + c["dur"]) - x0 - GAP)
        s.append(f'<rect x="{x0:.2f}" y="{yl1}" width="{wc:.2f}" height="{lane_h}" fill="{CAT1}"/>')
        # oracle slice, clipped so the 2 oracle-longer-than-compile cases don't overlap the neighbor
        wo_raw = X(c["ts"] + min(o["dur"], c["dur"])) - x0
        wo = max(0.6, wo_raw - GAP)
        s.append(f'<rect x="{x0:.2f}" y="{yl2}" width="{wo:.2f}" height="{lane_h}" fill="{CAT2}"/>')
        check(x0 >= ml - 0.01 and x0 + wc <= ml + pw + 1.0, f"fig3 slice {i} within plot")
        ratio = c["dur"] / o["dur"] if o["dur"] else float("nan")
        nxt = X(pairs[i + 1][0]["ts"]) if i + 1 < len(pairs) else ml + pw
        tip = json.dumps([f"{ratio:.2f}× above ceiling" if ratio >= 1 else f"{ratio:.2f}× (at/below ceiling)",
                          o["args"].get("node_name", c["name"]),
                          f"compile {c['dur']:,.0f} µs · oracle {o['dur']:,.0f} µs",
                          f"pattern {o['args'].get('pattern_hash','')} / shape {o['args'].get('shape_hash','')}"])
        hw = max(6.0, nxt - x0)
        hw = min(hw, W - x0)  # keep hit rect inside viewport
        hit_rects.append(f'<rect class="hit" x="{x0:.2f}" y="{yl1}" width="{hw:.2f}" height="{yl2 + lane_h - yl1}" '
                         f'data-tip="{esc(tip)}"/>')

    # direct label: largest-gap kernel (6.7x), inside its own headroom whitespace
    # (kernel name lives in the tooltip + table; inline text must not run over neighbors)
    bc, bo = biggest
    br = bc["dur"] / bo["dur"]
    bidx = pairs.index((bc, bo))
    next_start = X(pairs[bidx + 1][0]["ts"]) if bidx + 1 < len(pairs) else ml + pw
    lx = X(bc["ts"] + bo["dur"]) + 8
    lab = f"{br:.1f}× above ceiling"
    s.append(f'<text x="{lx:.2f}" y="{yl2 + lane_h / 2 + 4:.2f}" font-size="11.5" font-weight="650" fill="{INK}">{esc(lab)}</text>')
    check(lx + est_w(lab, 11.5) < next_start - 4, f"fig3 gap label stays inside its own whitespace: {lab}")

    # lane totals at right edge
    s.append(f'<text x="{ml + pw + 8}" y="{yl1 + lane_h / 2 - 2:.2f}" font-size="11.5" font-weight="650" fill="{INK}">{total_c:,.0f} µs</text>')
    s.append(f'<text x="{ml + pw + 8}" y="{yl1 + lane_h / 2 + 11:.2f}" font-size="10.5" fill="{INK2}">total</text>')
    s.append(f'<text x="{ml + pw + 8}" y="{yl2 + lane_h / 2 - 2:.2f}" font-size="11.5" font-weight="650" fill="{INK}">{total_o:,.0f} µs</text>')
    s.append(f'<text x="{ml + pw + 8}" y="{yl2 + lane_h / 2 + 11:.2f}" font-size="10.5" fill="{INK2}">{total_c / total_o:.1f}× headroom</text>')
    check(ml + pw + 8 + est_w(f"{total_c:,.0f} us", 11.5) <= W, "fig3 totals inside viewport")

    s.extend(hit_rects)  # hit layer on top
    s.append('</svg>')

    legend = (f'<div class="legend"><span><span class="sw" style="background:{CAT1}"></span>torch.compile (current)</span>'
              f'<span><span class="sw" style="background:{CAT2}"></span>reference-kernel ceiling (oracle)</span>'
              f'<span style="color:{MUTED}">hover any column for the kernel, both times, and the ratio</span></div>')
    foot = ("Caveat: idealized serial timeline from isolated per-kernel benches; reduction-kernel subset of "
            "mobilenetv2 (priced coverage). Extern slices (conv/GEMM/SDPA) omitted — identical in both lanes. "
            "Two of 46 kernels already sit at/below the oracle (ratio ≤ 1); their ceiling bars are clipped to the compile width.")
    trows = "".join(
        f"<tr><td>{esc(o['args'].get('node_name',''))}</td><td class='num'>{c['dur']:,.1f}</td>"
        f"<td class='num'>{o['dur']:,.1f}</td><td class='num'>{(c['dur']/o['dur']):.2f}×</td></tr>"
        for c, o in pairs)
    table = ("<details><summary>Data table (46 kernel pairs, execution order)</summary><table>"
             "<tr><th>kernel</th><th class='num'>compile µs</th><th class='num'>oracle µs</th>"
             "<th class='num'>ratio</th></tr>" + trows + "</table></details>")
    return page("One model’s kernels in execution order: current vs ceiling",
                "each bar = one fused kernel; the gap after each oracle bar is the remaining headroom",
                "".join(s), legend, foot, table, W + 54), total_c, total_o, (bo["args"]["node_name"], br, bc["dur"], bo["dur"])

# =====================================================================
INDEX = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Post figures</title>
<style>
  body {{ font-family:{font}; background:{page}; color:{ink}; max-width:760px; margin:48px auto; padding:0 24px; }}
  h1 {{ font-size:20px; }} li {{ margin:10px 0; font-size:15px; }}
  a {{ color:#1c5cab; }} .d {{ color:{ink2}; font-size:13px; }}
</style></head><body>
<h1>Perf-engineering post figures</h1>
<ol>
  <li><a href="fig1_per_commit.html">Fig 1 — Which commits carry the win (per-commit marginal delta)</a>
      <div class="d">30 named commits, kernel-geomean vs per-model e2e, aligned diverging panels</div></li>
  <li><a href="fig2_per_model.html">Fig 2 — 158 models, before → after (concentration)</a>
      <div class="d">sorted per-model e2e change, median {med:+.2f}%, ±0.82% noise band</div></li>
  <li><a href="fig3_timeline.html">Fig 3 — Kernels in execution order: current vs ceiling</a>
      <div class="d">46 kernel pairs, compile {tc:,.0f} µs vs oracle {to:,.0f} µs ({rx:.1f}×)</div></li>
</ol>
</body></html>
"""

def main():
    os.makedirs(OUT, exist_ok=True)
    h1 = fig1()
    h2, med, models = fig2()
    h3, tc, to, big = fig3()
    open(OUT + "/fig1_per_commit.html", "w").write(h1)
    open(OUT + "/fig2_per_model.html", "w").write(h2)
    open(OUT + "/fig3_timeline.html", "w").write(h3)
    open(OUT + "/index.html", "w").write(INDEX.format(font=FONT, page=PAGE, ink=INK, ink2=INK2,
                                                      med=med, tc=tc, to=to, rx=tc / to))
    print(f"OK: {len(ASSERTS)} geometry assertions passed")
    print(f"fig2 median {med:+.4f}%  top3: " + ", ".join(f"{k.split('/')[-1]} {v:+.2f}" for k, v in models[:3]))
    print(f"fig3 totals compile {tc:,.1f}us oracle {to:,.1f}us ratio {tc/to:.2f}x  biggest gap: {big[0]} {big[1]:.2f}x ({big[2]:,.1f} vs {big[3]:,.1f}us)")

if __name__ == "__main__":
    main()
