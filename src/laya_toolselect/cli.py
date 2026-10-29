"""Command line: laya-toolselect select TOOLS.json "request" [-k 5|auto] [--judge multilingual]

TOOLS.json is a JSON list of tool definitions in any supported format (OpenAI, Anthropic,
MCP), or an object with a "tools" list. Prints the selected tools as JSON.
"""
import argparse
import json
from pathlib import Path
import sys

from . import LayaJudge, ToolSelector, __version__


def _load_tools(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tools"] if isinstance(data, dict) else data


def main(argv=None):
    p = argparse.ArgumentParser(prog="laya-toolselect", description="Pick the tools an LLM needs for a request")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="select tools for one request")
    s.add_argument("tools", type=Path)
    s.add_argument("query")
    s.add_argument("-k", default="5", help='number of tools, or "auto"')
    s.add_argument("--format", choices=["openai", "anthropic", "mcp", "names"], default="names")
    s.add_argument("--examples", type=Path, help="JSON {tool_name: [example requests]} from save_examples()")
    s.add_argument("--no-semantic", action="store_true", help="BM25 only (no wordllama)")
    s.add_argument("--judge", choices=["english", "multilingual"], help="add the Laya judge")
    s.add_argument("--device", default=None)
    s.add_argument("--explain", action="store_true", help="print per-tool scores to stderr")
    a = p.parse_args(argv)

    judge = LayaJudge(a.judge, device=a.device) if a.judge else None
    sel = ToolSelector(_load_tools(a.tools), embedder=None if a.no_semantic else "auto", judge=judge)
    if a.examples:
        sel.load_examples(a.examples)
    k = "auto" if a.k == "auto" else int(a.k)
    res = sel.select(a.query, k=k)
    if a.explain:
        for c in res.shortlist[:10]:
            print(f"{c.name:40} score={c.score:.4f} lexical={c.lexical:.2f} semantic={c.semantic:.3f}"
                  + (f" laya={c.judge:.3f}" if c.judge is not None else ""), file=sys.stderr)
        if res.p_none is not None:
            print(f"p(no tool fits)={res.p_none:.3f}", file=sys.stderr)
        print(f"{len(res)}/{res.total_tools} tools, {res.savings:.1%} of schema text saved, "
              f"{res.latency_ms:.1f} ms", file=sys.stderr)
    out = {"names": lambda: res.names, "openai": res.as_openai, "anthropic": res.as_anthropic,
           "mcp": res.as_mcp}[a.format]()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
