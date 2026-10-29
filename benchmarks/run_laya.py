"""Measure what the Laya judge adds. Needs `pip install 'laya-toolselect[laya,semantic]'` and
access to huggingface.co (first run downloads the checkpoint).

    python benchmarks/run_laya.py --data benchmarks/data --n 300 --device mps   # Apple Silicon
    python benchmarks/run_laya.py --data benchmarks/data --n 300 --device cuda

Reports, on the test split (settings were fixed on dev before this):
  * Recall@1/@3 and MRR of hybrid retrieval alone vs hybrid + Laya (fuse and judge modes)
  * k="auto": tools exposed and recall, with and without Laya
  * latency per request
  * no-tool detection: AUROC of Laya's p_none on BFCL irrelevance vs simple questions
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from laya_toolselect import LayaJudge, ToolSelector  # noqa: E402


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def run(sel, queries, use_judge):
    r1, r3, rr, n_auto, hit_auto, ms = [], [], [], [], [], []
    for q in queries:
        gold = set(q["gold"])
        t = time.perf_counter()
        res = sel.select(q["query"], k=10, use_judge=use_judge)
        ms.append(1000 * (time.perf_counter() - t))
        names = res.names
        r1.append(names[0] in gold if names else False)
        r3.append(any(n in gold for n in names[:3]))
        rr.append(next((1 / (i + 1) for i, n in enumerate(names) if n in gold), 0.0))
        auto = sel.select(q["query"], k="auto", use_judge=use_judge)
        n_auto.append(len(auto))
        hit_auto.append(any(n in gold for n in auto.names))
    return {"R@1": np.mean(r1), "R@3": np.mean(r3), "MRR@10": np.mean(rr), "auto_tools": np.mean(n_auto),
            "auto_recall": np.mean(hit_auto), "ms/request": np.mean(ms)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    p.add_argument("--n", type=int, default=300, help="test queries per dataset")
    p.add_argument("--models", nargs="+", default=["multilingual", "english"])
    p.add_argument("--device", default=None)
    p.add_argument("--learn", type=int, default=0, help="MetaTool usage-log examples per tool")
    p.add_argument("--out", type=Path, default=Path("laya_results.json"))
    a = p.parse_args()
    results = {}
    for model in a.models:
        judge = LayaJudge(model, device=a.device)
        for ds in ["metatool", "bfcl"]:
            d = json.loads((a.data / f"{ds}.json").read_text())
            qs = random.Random(0).sample(d["test"], min(a.n, len(d["test"])))
            for mode in ["fuse", "judge"]:
                sel = ToolSelector(d["tools"], judge=judge, judge_mode=mode)
                if a.learn and d["log"]:
                    per, pairs = {}, []
                    for it in d["log"]:
                        nm = it["gold"][0]
                        if per.get(nm, 0) < a.learn:
                            per[nm] = per.get(nm, 0) + 1
                            pairs.append((it["query"], nm))
                    sel.learn_many(pairs)
                if mode == "fuse":
                    results[f"{ds} | hybrid only"] = run(sel, qs, use_judge=False)
                    print(ds, "hybrid only", results[f"{ds} | hybrid only"], flush=True)
                key = f"{ds} | +laya-{model} ({mode})"
                results[key] = run(sel, qs, use_judge=True)
                print(key, results[key], flush=True)
        # no-tool detection
        nt = json.loads((a.data / "notool.json").read_text())
        p_irr, p_rel = [], []
        for item in nt:
            sel = ToolSelector(item["functions"], judge=judge, embedder=None)
            res = sel.select(item["query"], k=1)
            (p_rel if item["tool_fits"] else p_irr).append(res.p_none)
        results[f"notool | laya-{model}"] = {
            "AUROC(p_none)": auroc(p_irr, p_rel), "mean p_none irrelevant": float(np.mean(p_irr)),
            "mean p_none relevant": float(np.mean(p_rel))}
        print("notool", model, results[f"notool | laya-{model}"], flush=True)
    a.out.write_text(json.dumps({k: {m: float(v) for m, v in r.items()} for k, r in results.items()}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
