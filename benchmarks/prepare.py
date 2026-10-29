"""Build the two benchmark files from public GitHub sources.

MetaTool (Huang et al., ICLR 2024; github.com/HowieHwong/MetaTool):
  dataset/plugin_des.json, dataset/data/all_clean_data.csv -> 199 tools, 20,614 (query, tool) pairs.
  Per tool, half the queries are a "usage log" (used only by the learn() experiments), the other
  half are evaluation queries, from which dev (tuning) and test (reporting) are sampled.

BFCL v4 (Berkeley Function Calling Leaderboard; github.com/ShishirPatil/gorilla):
  simple_python, multiple, live_simple, live_multiple + possible_answer. Every function from every
  query is pooled into ONE catalog, so each request competes against all of them. Only single-call
  questions are kept. A real catalog has one tool per name, so functions are keyed by name and
  the first definition seen is kept (BFCL repeats some names with reworded descriptions).

Usage: python benchmarks/prepare.py --metatool DIR --bfcl DIR --out benchmarks/data
"""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import random


def metatool(src: Path, rng: random.Random):
    des = json.loads((src / "plugin_des.json").read_text())
    rows = list(csv.DictReader((src / "all_clean_data.csv").open(encoding="utf-8")))
    tools = [{"name": n, "description": d} for n, d in sorted(des.items())]
    by_tool = defaultdict(list)
    for r in rows:
        by_tool[r["Tool"]].append(r["Query"].strip())
    log, evalq = [], []
    for name in sorted(by_tool):
        qs = sorted(set(by_tool[name]))
        rng.shuffle(qs)
        half = len(qs) // 2
        log += [{"query": q, "gold": [name]} for q in qs[:half]]
        evalq += [{"query": q, "gold": [name]} for q in qs[half:]]
    rng.shuffle(evalq)
    return {"name": "MetaTool", "tools": tools, "log": log,
            "dev": evalq[:500], "test": evalq[500:2500]}


def bfcl(src: Path, rng: random.Random):
    tools, key2id, names = [], {}, defaultdict(int)
    queries = []
    for cat in ["simple_python", "multiple", "live_simple", "live_multiple"]:
        answers = {}
        for line in (src / f"ans_{cat}.json").read_text().splitlines():
            if line.strip():
                a = json.loads(line)
                answers[a["id"]] = a["ground_truth"]
        for line in (src / f"BFCL_v4_{cat}.json").read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            local = {}
            for f in item["function"]:
                if f["name"] not in key2id:
                    key2id[f["name"]] = f["name"]
                    tools.append({"name": f["name"], "description": f.get("description", ""),
                                  "parameters": f.get("parameters", {})})
                else:
                    names[f["name"]] += 1
                local[f["name"]] = f["name"]
            gt = answers.get(item["id"])
            if not gt or len(gt) != 1:
                continue
            gold_name = next(iter(gt[0]))
            if gold_name not in local:
                continue
            turns = item["question"][0] if item["question"] and isinstance(item["question"][0], list) else item["question"]
            user = [t["content"] for t in turns if t.get("role") == "user"]
            if not user:
                continue
            queries.append({"query": user[-1], "gold": [local[gold_name]], "category": cat})
    rng.shuffle(queries)
    n_dev = len(queries) * 3 // 10
    repeat_definitions = sum(names.values())
    return {"name": "BFCL-pooled", "tools": tools, "log": [], "dev": queries[:n_dev],
            "test": queries[n_dev:], "repeat_definitions": repeat_definitions}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metatool", type=Path, required=True)
    p.add_argument("--bfcl", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    # No-tool detection set: BFCL irrelevance (none of the offered functions fits) vs
    # simple_python (the one offered function fits), each judged over its own function list.
    nt = []
    for cat, fits in [("irrelevance", False), ("simple_python", True)]:
        for line in (a.bfcl / f"BFCL_v4_{cat}.json").read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                turns = item["question"][0]
                user = [t["content"] for t in turns if t.get("role") == "user"]
                nt.append({"query": user[-1], "functions": item["function"], "tool_fits": fits})
    (a.out / "notool.json").write_text(json.dumps(nt, ensure_ascii=False))
    print("notool", sum(not x["tool_fits"] for x in nt), "irrelevant /", sum(x["tool_fits"] for x in nt), "relevant")
    for name, fn, src in [("metatool", metatool, a.metatool), ("bfcl", bfcl, a.bfcl)]:
        d = fn(src, random.Random(a.seed))
        (a.out / f"{name}.json").write_text(json.dumps(d, ensure_ascii=False))
        print(name, "tools", len(d["tools"]), "log", len(d["log"]), "dev", len(d["dev"]),
              "test", len(d["test"]), "repeated definitions dropped", d.get("repeat_definitions", 0))


if __name__ == "__main__":
    main()
