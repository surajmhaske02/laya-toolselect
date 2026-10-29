"""Evaluate tool selection on MetaTool and pooled BFCL.

    python benchmarks/run.py --data benchmarks/data --split dev   # choose settings
    python benchmarks/run.py --data benchmarks/data --split test  # report

Metrics per query: whether the gold tool is in the top 1/3/5/10, reciprocal rank (MRR@10),
and for k="auto" the number of tools exposed and recall. "Schema share" is the fraction of
tool-definition characters sent, versus sending the whole catalog.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from laya_toolselect import ToolSelector  # noqa: E402


def evaluate(selector, queries, auto=True):
    hits = {1: [], 3: [], 5: [], 10: []}
    rr, lat, auto_n, auto_hit, auto_chars, auto_share = [], [], [], [], [], []
    total_chars = None
    for q in queries:
        gold = set(q["gold"])
        t = time.perf_counter()
        ranked = selector.retrieve(q["query"], 10)
        lat.append(1000 * (time.perf_counter() - t))
        names = [c.name for c in ranked]
        for k in hits:
            hits[k].append(any(n in gold for n in names[:k]))
        rr.append(next((1 / (i + 1) for i, n in enumerate(names) if n in gold), 0.0))
        if auto:
            sel = selector.select(q["query"], k="auto")
            auto_n.append(len(sel))
            auto_hit.append(any(n in gold for n in sel.names))
            auto_share.append(sel.schema_chars / sel.total_schema_chars)
    out = {f"R@{k}": float(np.mean(v)) for k, v in hits.items()}
    out["MRR@10"] = float(np.mean(rr))
    out["ms/query"] = float(np.mean(lat))
    if auto:
        out.update({"auto_tools": float(np.mean(auto_n)), "auto_recall": float(np.mean(auto_hit)),
                    "auto_schema_share": float(np.mean(auto_share))})
    return out


CONFIGS = {
    "bm25-desc-only":   dict(embedder=None, name_weight=0, use_params=False, use_stem=False),
    "bm25":             dict(embedder=None),
    "bm25-nostem":      dict(embedder=None, use_stem=False),
    "semantic":         dict(embedder="wordllama", fusion="weighted", alpha=0.0),
    "hybrid-w0.3":      dict(embedder="wordllama", fusion="weighted", alpha=0.3),
    "hybrid-w0.5":      dict(embedder="wordllama", fusion="weighted", alpha=0.5),
    "hybrid-w0.7":      dict(embedder="wordllama", fusion="weighted", alpha=0.7),
    "hybrid-rrf":       dict(embedder="wordllama", fusion="rrf"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--learn", nargs="*", type=int, default=[0],
                   help="usage-log examples per tool to learn() first (MetaTool only)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    results = {}
    for ds in ["metatool", "bfcl"]:
        d = json.loads((a.data / f"{ds}.json").read_text())
        queries = d[a.split]
        for cfg in a.configs:
            for n_ex in (a.learn if d["log"] else [0]):
                sel = ToolSelector(d["tools"], **CONFIGS[cfg])
                if n_ex:
                    per = {}
                    pairs = []
                    for item in d["log"]:
                        name = item["gold"][0]
                        if per.get(name, 0) < n_ex:
                            per[name] = per.get(name, 0) + 1
                            pairs.append((item["query"], name))
                    sel.learn_many(pairs)
                t = time.perf_counter()
                sel.index()
                build = time.perf_counter() - t
                r = evaluate(sel, queries)
                r["index_s"] = build
                key = f"{d['name']} | {cfg}" + (f" | learn {n_ex}/tool" if n_ex else "")
                results[key] = r
                print(f"{key:45} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
