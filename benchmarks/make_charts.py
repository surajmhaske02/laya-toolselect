"""Draw the README charts (plain SVG, no dependencies) from benchmarks/results/test.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "benchmarks/results/test.json").read_text())
OUT = ROOT / "docs/assets"
BLUE, GREY, LIGHT, TEXT, SUB, GRID = "#2f6fdb", "#9aa0a6", "#c9cdd2", "#1f2328", "#59636e", "#e6e8eb"
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"


def bars():
    groups = [("MetaTool · 199 tools", [("Plain keyword search", R["MetaTool | bm25-desc-only"]["R@5"], LIGHT),
                                        ("laya-toolselect", R["MetaTool | hybrid-w0.3"]["R@5"], BLUE),
                                        ("+ learned usage", R["MetaTool | hybrid-w0.3 | learn 50/tool"]["R@5"], "#173f8a")]),
              ("BFCL · 1,096 tools", [("Plain keyword search", R["BFCL-pooled | bm25-desc-only"]["R@5"], LIGHT),
                                      ("laya-toolselect", R["BFCL-pooled | hybrid-w0.3"]["R@5"], BLUE)])]
    W, H, x0, y0, base = 760, 330, 40, 90, 280
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         f"<text x='{x0}' y='36' font-size='19' font-weight='600' fill='{TEXT}'>The right tool reaches the LLM far more often</text>",
         f"<text x='{x0}' y='60' font-size='13' fill='{SUB}'>Share of requests whose correct tool is among the 5 tools sent (held-out test sets)</text>",
         f"<line x1='{x0}' x2='{W-20}' y1='{base}' y2='{base}' stroke='{GREY}'/>"]
    bw, gap, gx = 104, 18, x0 + 4
    for title, items in groups:
        for label, v, col in items:
            h = v * (base - y0)
            s.append(f"<rect x='{gx}' y='{base-h:.1f}' width='{bw}' height='{h:.1f}' rx='3' fill='{col}'/>")
            s.append(f"<text x='{gx+bw/2}' y='{base-h-8:.1f}' text-anchor='middle' font-size='15' font-weight='600' fill='{TEXT}'>{v:.0%}</text>")
            s.append(f"<text x='{gx+bw/2}' y='{base+18}' text-anchor='middle' font-size='11.5' fill='{SUB}'>{label}</text>")
            gx += bw + gap
        s.append(f"<text x='{gx - (bw+gap)*len(items)/2 - gap/2}' y='{base+40}' text-anchor='middle' font-size='13' font-weight='600' fill='{TEXT}'>{title}</text>")
        gx += 40
    s.append("</svg>")
    (OUT / "recall_at5.svg").write_text("\n".join(s))


def curve():
    series = [("Keyword (BM25)", "bm25", GREY, ""), ("Semantic", "semantic", GREY, "5 4"), ("laya-toolselect (hybrid)", "hybrid-w0.3", BLUE, "")]
    ns = [0, 5, 20, 50]
    W, H, left, right, y0, base = 760, 340, 70, 520, 80, 290
    x = lambda i: left + i * (right - left) / 3
    y = lambda v: base - (v - 0.5) / 0.5 * (base - y0)
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         f"<text x='30' y='34' font-size='19' font-weight='600' fill='{TEXT}'>Learning from past requests: 75% → 91%</text>",
         f"<text x='30' y='58' font-size='13' fill='{SUB}'>MetaTool test set: right tool among the 5 sent, after selector.learn() on N past requests per tool</text>"]
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        s.append(f"<line x1='{left}' x2='{right}' y1='{y(t):.1f}' y2='{y(t):.1f}' stroke='{GRID}'/>")
        s.append(f"<text x='{left-10}' y='{y(t)+4:.1f}' text-anchor='end' font-size='12' fill='{SUB}'>{t:.0%}</text>")
    for i, n in enumerate(ns):
        s.append(f"<text x='{x(i):.1f}' y='{base+22}' text-anchor='middle' font-size='12' fill='{SUB}'>{'none' if n == 0 else n}</text>")
    s.append(f"<text x='{(left+right)/2}' y='{base+44}' text-anchor='middle' font-size='12.5' fill='{SUB}'>Past requests learned per tool</text>")
    ends = []
    for name, key, col, dash in series:
        vals = [R[f"MetaTool | {key}" + (f" | learn {n}/tool" if n else "")]["R@5"] for n in ns]
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        w = 3.5 if col == BLUE else 2
        s.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='{w}'" + (f" stroke-dasharray='{dash}'" if dash else "") + "/>")
        for i, v in enumerate(vals):
            s.append(f"<circle cx='{x(i):.1f}' cy='{y(v):.1f}' r='{4.5 if col == BLUE else 3.5}' fill='{col}'/>")
        ends.append((vals[-1], name, col))
    prev = -1e9
    for v, name, col in sorted(ends, reverse=True):
        yy = max(y(v) + 4, prev + 20)
        prev = yy
        s.append(f"<text x='{right+14}' y='{yy:.1f}' font-size='13' font-weight='{600 if col == BLUE else 400}' fill='{col if col == BLUE else SUB}'>{name} {v:.1%}</text>")
    s.append("</svg>")
    (OUT / "learning_curve.svg").write_text("\n".join(s))


if __name__ == "__main__":
    bars()
    curve()
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
