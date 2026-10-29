"""ToolSelector: pick the few tools an LLM should see for one request.

Pipeline (each stage optional except the first):

1. Lexical retrieval -- BM25 over tool name (split camelCase/snake_case), description,
   parameter names/descriptions and learned example requests.
2. Semantic retrieval -- embedding similarity (default: offline WordLlama), with
   max-similarity over learned example requests.
3. Fusion of 1 and 2 (weighted min-max by default, or RRF).
4. Laya judge -- one Laya forward pass over the top ``shortlist`` tools, plus a ``none``
   option, fused with the retrieval ranking.
5. ``k="auto"`` -- expose as few tools as the scores justify, between ``min_k`` and ``max_k``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

from .bm25 import BM25
from .embed import resolve_embedder
from .text import split_identifier, tokenize
from .tools import Tool, to_tool, to_tools

FUSIONS = ("weighted", "rrf")


@dataclass
class Candidate:
    tool: Tool
    index: int                 # position in the selector's tool list
    score: float               # final score the list is sorted by
    lexical: float = 0.0       # BM25 score
    semantic: float = 0.0      # cosine similarity (max over description and examples)
    judge: Optional[float] = None  # Laya probability, when a judge ran
    confidence: float = 0.0    # retrieval score min-max normalised to [0, 1] (used by k="auto")

    @property
    def name(self) -> str:
        return self.tool.name


@dataclass
class Selection:
    query: str
    candidates: list                         # selected, best first
    shortlist: list = field(default_factory=list)  # everything that was scored, best first
    p_none: Optional[float] = None            # Laya's "no listed tool fits" probability
    total_tools: int = 0
    total_schema_chars: int = 0
    latency_ms: float = 0.0

    @property
    def tools(self) -> list:
        """The selected tools in the format they were registered in (same objects)."""
        return [c.tool.raw if c.tool.raw is not None else c.tool for c in self.candidates]

    @property
    def names(self) -> list:
        return [c.tool.name for c in self.candidates]

    @property
    def schema_chars(self) -> int:
        return sum(c.tool.schema_chars() for c in self.candidates)

    @property
    def savings(self) -> float:
        """Fraction of tool-definition characters not sent to the LLM."""
        return 0.0 if not self.total_schema_chars else 1 - self.schema_chars / self.total_schema_chars

    def as_openai(self) -> list:
        return [c.tool.to_openai() for c in self.candidates]

    def as_anthropic(self) -> list:
        return [c.tool.to_anthropic() for c in self.candidates]

    def as_mcp(self) -> list:
        return [c.tool.to_mcp() for c in self.candidates]

    def __iter__(self):
        return iter(self.tools)

    def __len__(self):
        return len(self.candidates)


class ToolSelector:
    def __init__(self, tools: Iterable[Any] = (), *, embedder: Any = "auto", judge: Any = None,
                 fusion: str = "weighted", alpha: float = 0.3, rrf_k: int = 60,
                 name_weight: int = 2, use_params: bool = True, use_stem: bool = True,
                 shortlist: int = 10, judge_mode: str = "fuse", judge_weight: float = 0.5,
                 max_examples_per_tool: int = 200):
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if judge_mode not in ("fuse", "judge"):
            raise ValueError("judge_mode must be 'fuse' or 'judge'")
        if type(shortlist) is not int or shortlist < 1:
            raise ValueError("shortlist must be a positive integer")
        self.embedder = resolve_embedder(embedder)
        self.judge = judge
        self.fusion, self.alpha, self.rrf_k = fusion, alpha, rrf_k
        self.name_weight, self.use_params, self.use_stem = name_weight, use_params, use_stem
        self.shortlist, self.judge_mode, self.judge_weight = shortlist, judge_mode, judge_weight
        self.max_examples_per_tool = max_examples_per_tool
        self._tools: list = []
        self._by_name: dict = {}
        self._lock = threading.RLock()
        self._dirty = True
        self._desc_vecs = None
        self._ex_vecs = None
        self._ex_owner = None
        self._vec_cache: dict = {}
        self.add_tools(tools)

    # ------------------------------------------------------------------ registry
    @property
    def tools(self) -> list:
        return list(self._tools)

    def __len__(self):
        return len(self._tools)

    def add_tools(self, tools: Iterable[Any]) -> "ToolSelector":
        """Register tools. A tool whose name is already registered replaces the old one."""
        with self._lock:
            for t in to_tools(tools):
                if t.name in self._by_name:
                    old = self._by_name[t.name]
                    t.examples = list(dict.fromkeys(old.examples + t.examples))
                    self._tools[self._tools.index(old)] = t
                else:
                    self._tools.append(t)
                self._by_name[t.name] = t
            self._dirty = True
        return self

    def add_tool(self, tool: Any) -> "ToolSelector":
        return self.add_tools([tool])

    def remove_tools(self, names: Iterable[str]) -> "ToolSelector":
        with self._lock:
            names = set(names)
            self._tools = [t for t in self._tools if t.name not in names]
            self._by_name = {t.name: t for t in self._tools}
            self._dirty = True
        return self

    def learn(self, query: str, tool_name: str) -> None:
        """Record that ``query`` was correctly served by ``tool_name`` (usage log / feedback).

        Learned requests are indexed like extra documentation for the tool and are the
        single biggest accuracy lever measured (see the benchmark report).
        """
        self.learn_many([(query, tool_name)])

    def learn_many(self, pairs: Iterable[tuple]) -> None:
        with self._lock:
            for query, name in pairs:
                tool = self._by_name.get(name)
                if tool is None:
                    raise KeyError(f"unknown tool {name!r}")
                if query and query not in tool.examples:
                    tool.examples.append(str(query))
                    if len(tool.examples) > self.max_examples_per_tool:
                        tool.examples.pop(0)
            self._dirty = True

    def save_examples(self, path: Union[str, Path]) -> None:
        data = {t.name: t.examples for t in self._tools if t.examples}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def load_examples(self, path: Union[str, Path]) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.learn_many((q, name) for name, qs in data.items() if name in self._by_name for q in qs)

    # ------------------------------------------------------------------ index
    def _doc_tokens(self, t: Tool) -> list:
        toks = tokenize(split_identifier(t.name), use_stem=self.use_stem) * self.name_weight
        toks += tokenize(t.description, use_stem=self.use_stem)
        if self.use_params:
            toks += tokenize(t.param_text(), use_stem=self.use_stem)
        for e in t.examples:
            toks += tokenize(e, use_stem=self.use_stem)
        toks += tokenize(" ".join(t.tags), use_stem=self.use_stem)
        return toks

    def _embed(self, texts: list) -> np.ndarray:
        """Embed with a per-text cache, so re-indexing after add/learn is incremental."""
        missing = [s for s in dict.fromkeys(texts) if s not in self._vec_cache]
        if missing:
            for s, v in zip(missing, self.embedder(missing)):
                self._vec_cache[s] = v
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self._vec_cache[s] for s in texts])

    def _build(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._bm25 = BM25([self._doc_tokens(t) for t in self._tools])
            if self.embedder is not None and self._tools:
                self._desc_vecs = self._embed([t.card() for t in self._tools])
                owners, texts = [], []
                for i, t in enumerate(self._tools):
                    for e in t.examples:
                        owners.append(i)
                        texts.append(e)
                self._ex_owner = np.array(owners, dtype=int)
                self._ex_vecs = self._embed(texts) if texts else None
            self._dirty = False

    def index(self) -> "ToolSelector":
        """Build the index now (otherwise it is built lazily on the first query)."""
        self._build()
        return self

    # ------------------------------------------------------------------ retrieval
    def _semantic(self, query: str) -> Optional[np.ndarray]:
        if self.embedder is None or not self._tools:
            return None
        qv = self.embedder([query])[0]
        sim = self._desc_vecs @ qv
        if self._ex_vecs is not None and len(self._ex_owner):
            ex = self._ex_vecs @ qv
            best = np.full(len(self._tools), -1.0, dtype=np.float32)
            np.maximum.at(best, self._ex_owner, ex)
            sim = np.maximum(sim, best)
        return sim

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)

    def _confidence(self, lex: np.ndarray, sem: Optional[np.ndarray]) -> np.ndarray:
        if sem is None:
            return self._minmax(lex)
        return self.alpha * self._minmax(lex) + (1 - self.alpha) * self._minmax(sem)

    def _fused(self, lex: np.ndarray, sem: Optional[np.ndarray]) -> np.ndarray:
        if sem is None:
            return lex
        if self.fusion == "rrf":
            r = np.empty(len(lex))
            out = np.zeros(len(lex))
            for s in (lex, sem):
                r[np.argsort(-s, kind="stable")] = np.arange(len(s))
                out += 1.0 / (self.rrf_k + 1 + r)
            return out
        return self.alpha * self._minmax(lex) + (1 - self.alpha) * self._minmax(sem)

    def retrieve(self, query: str, n: Optional[int] = None) -> list:
        """First-stage ranking only (no judge). Returns up to ``n`` candidates, best first."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        self._build()
        if not self._tools:
            return []
        lex = self._bm25.scores(tokenize(query, use_stem=self.use_stem))
        sem = self._semantic(query)
        fused = self._fused(lex, sem)
        conf = self._confidence(lex, sem)
        order = np.argsort(-fused, kind="stable")[: n or len(self._tools)]
        return [Candidate(self._tools[i], int(i), float(fused[i]), float(lex[i]),
                          float(sem[i]) if sem is not None else 0.0, confidence=float(conf[i]))
                for i in order]

    # ------------------------------------------------------------------ selection
    def select(self, query: str, k: Union[int, str] = 5, *, context: Optional[str] = None,
               min_k: int = 1, max_k: int = 8, coverage: float = 0.9,
               always_include: Sequence[str] = (), use_judge: Optional[bool] = None) -> Selection:
        """Pick the tools to show the LLM for ``query``.

        k: an int, or ``"auto"`` -- with a judge, the smallest set covering ``coverage`` of
           Laya's tool probability; without one, tools scoring within the auto margin of the best.
        context: recent conversation text, passed to the judge (retrieval uses ``query``).
        always_include: tool names exposed on every call (they do not count toward k).
        """
        t0 = time.perf_counter()
        if k != "auto" and (type(k) is not int or k < 0):
            raise ValueError("k must be a nonnegative int or 'auto'")
        if not 1 <= min_k <= max_k:
            raise ValueError("need 1 <= min_k <= max_k")
        run_judge = self.judge is not None if use_judge is None else (use_judge and self.judge is not None)
        width = max(self.shortlist if run_judge else 0, max_k if k == "auto" else k)
        shortlist = self.retrieve(query, width)
        p_none = None
        if run_judge and shortlist:
            head = shortlist[: self.shortlist]
            res = self.judge.score(query, [c.tool for c in head], context=context)
            for c, p in zip(head, res["probs"]):
                c.judge = p
            p_none = res["p_none"]
            n = len(head)
            judge_rank = np.empty(n)
            judge_rank[np.argsort(-np.array(res["probs"]), kind="stable")] = np.arange(n)
            for r, c in enumerate(head):
                if self.judge_mode == "judge":
                    c.score = c.judge
                else:  # weighted reciprocal-rank fusion of retrieval rank and judge rank
                    c.score = ((1 - self.judge_weight) / (self.rrf_k + 1 + r)
                               + self.judge_weight / (self.rrf_k + 1 + judge_rank[r]))
            floor = min(c.score for c in head)
            for j, c in enumerate(shortlist[n:]):
                c.score = floor - 1 - j
            shortlist.sort(key=lambda c: -c.score)

        if k == "auto":
            chosen = self._auto(shortlist, run_judge, min_k, max_k, coverage)
        else:
            chosen = shortlist[:k]
        names = {c.tool.name for c in chosen}
        for name in always_include:
            t = self._by_name.get(name)
            if t is not None and name not in names:
                chosen.append(Candidate(t, self._tools.index(t), float("nan")))
                names.add(name)
        return Selection(query=query, candidates=chosen, shortlist=shortlist, p_none=p_none,
                         total_tools=len(self._tools),
                         total_schema_chars=sum(t.schema_chars() for t in self._tools),
                         latency_ms=1000 * (time.perf_counter() - t0))

    #: Without a judge, ``k="auto"`` keeps tools whose fused score is within this margin of
    #: the best one. Chosen on the MetaTool/BFCL dev splits (see benchmarks/REPORT.md).
    auto_margin = 0.3

    def _auto(self, ranked, run_judge, min_k, max_k, coverage):
        if not ranked:
            return []
        if run_judge and ranked[0].judge is not None:
            probs = [c.judge or 0.0 for c in ranked]
            total = sum(probs) or 1.0
            out, acc = [], 0.0
            for c, p in zip(ranked, probs):
                if len(out) >= max_k or (len(out) >= min_k and acc >= coverage):
                    break
                out.append(c)
                acc += p / total
            return out
        top = max(c.confidence for c in ranked)
        out = [c for c in ranked[:max_k] if c.confidence >= top - self.auto_margin]
        return out if len(out) >= min_k else ranked[:min_k]

    # ------------------------------------------------------------------ meta-tool
    def search_tool(self, fmt: str = "openai", name: str = "search_tools") -> dict:
        """A tool definition the LLM can call to discover more tools on demand."""
        spec = Tool(name=name, description=(
            f"Search the catalog of {len(self._tools)} available tools. Call this when none of the "
            "tools you currently have fits the task; describe what you need to do."),
            parameters={"type": "object", "properties": {
                "query": {"type": "string", "description": "What you need a tool to do"},
                "k": {"type": "integer", "description": "How many tools to return", "default": 5}},
                "required": ["query"]})
        return {"openai": spec.to_openai, "anthropic": spec.to_anthropic, "mcp": spec.to_mcp}[fmt]()

    def handle_search(self, arguments: Union[str, dict], fmt: str = "openai") -> list:
        """Run a ``search_tools`` call and return the found tools' definitions in ``fmt``."""
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        k = int(args.get("k", 5) or 5)
        sel = self.select(str(args["query"]), k=max(1, min(k, 20)))
        return {"openai": sel.as_openai, "anthropic": sel.as_anthropic, "mcp": sel.as_mcp}[fmt]()
