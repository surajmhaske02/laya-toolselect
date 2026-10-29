# laya-toolselect reference

## ToolSelector(tools=(), **options)

`tools` accepts any mix of these formats:
- OpenAI Chat (`{"type":"function","function":{...}}`) and OpenAI Responses
- Anthropic (`input_schema`)
- MCP (`inputSchema`, `mcp.types.Tool`, or a whole `ListToolsResult`)
- LangChain tools
- plain Python functions (name, docstring, signature)
- `Tool` objects

| Option | Default | What it does |
|---|---|---|
| `embedder` | `"auto"` | `"auto"` = WordLlama if installed, else keyword only. Also accepts `"wordllama"`, a sentence-transformers model id, any callable `texts -> array`, or `None` |
| `judge` | `None` | A `LayaJudge`, adding Laya re-scoring and `p_none` |
| `alpha` | `0.3` | Keyword weight in hybrid fusion; semantic gets `1 - alpha` |
| `fusion` | `"weighted"` | `"weighted"` or `"rrf"` (reciprocal-rank fusion) |
| `shortlist` | `10` | How many top tools the judge scores |
| `judge_mode` | `"fuse"` | `"fuse"` blends Laya's and retrieval's rankings; `"judge"` sorts by Laya only |
| `judge_weight` | `0.5` | Laya's share in `"fuse"` mode |
| `name_weight` | `2` | How many times name words count in the keyword index |
| `use_params` | `True` | Index parameter names and descriptions |
| `max_examples_per_tool` | `200` | Learned requests kept per tool (oldest dropped first) |

| Method | Returns | Notes |
|---|---|---|
| `select(query, k=5, *, context=None, min_k=1, max_k=8, coverage=0.9, always_include=(), use_judge=None)` | `Selection` | `k` is an int or `"auto"`. `context` (recent conversation) goes to the judge. `always_include` names are added on top of `k` |
| `retrieve(query, n=None)` | `list[Candidate]` | Retrieval only (no judge), best first |
| `learn(query, tool_name)` / `learn_many(pairs)` | — | Adds requests to a tool's examples. Raises `KeyError` for an unknown tool |
| `save_examples(path)` / `load_examples(path)` | — | JSON `{tool_name: [requests]}`, to keep learning across restarts |
| `add_tools(tools)` / `add_tool(tool)` / `remove_tools(names)` | `self` | A tool with an existing name replaces the old one and keeps its examples |
| `index()` | `self` | Builds the index now; otherwise it is built on the first query |
| `search_tool(fmt="openai")` | `dict` | A `search_tools` meta-tool definition (`"openai"`, `"anthropic"` or `"mcp"`) |
| `handle_search(arguments, fmt="openai")` | `list[dict]` | Runs a `search_tools` call (JSON string or dict) and returns the found tools' definitions |

`k="auto"` without a judge keeps every tool whose normalized score is within `auto_margin`
(0.3) of the best one, between `min_k` and `max_k` tools. With a judge, it keeps the fewest
tools covering `coverage` of Laya's probability.

## Selection

| Field | Meaning |
|---|---|
| `tools` | Selected tools exactly as registered, best first |
| `names` | Their names |
| `candidates` | `Candidate` objects with `score`, `lexical`, `semantic`, `confidence` (0–1) and `judge` (Laya probability or `None`) |
| `shortlist` | Every candidate that was scored, best first |
| `p_none` | Laya's probability that no listed tool fits (`None` without a judge) |
| `savings` | Share of the catalog's tool-definition characters not sent |
| `latency_ms` | Time spent in `select` |
| `as_openai()` / `as_anthropic()` / `as_mcp()` | The selection converted to that format |

## LayaJudge(model="multilingual", **options)

The top `shortlist` tools plus a `none` option become one Laya `choice` question. With
`debias=True`, a second copy with the tools reversed runs in the same forward pass, and the two
results are averaged, which cancels position bias. Option text is sized to fit Laya's token
budget.

| Option | Default | What it does |
|---|---|---|
| `model` | `"multilingual"` | `"multilingual"` (mmBERT-base, 1,024 tokens, 100+ languages) or `"english"` (ModernBERT-large, 512 tokens), or a checkpoint path |
| `device` | auto | `"cpu"`, `"cuda"` or `"mps"` |
| `revision` | `None` | Pins a Hugging Face commit |
| `debias` | `True` | Adds the reversed-order question in the same forward pass |
| `query_tokens` | `96` | Tokens reserved for the request; the rest of the budget goes to tool options |
| `agent` | `None` | An already loaded `laya.load(...)` agent |

## Embedders

- `WordLlamaEmbedder()`: the default. Offline; the weights ship in the `wordllama` package.
- `SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")`: any sentence-transformers model.
- `CallableEmbedder(fn)`: wraps any function that maps `list[str]` to an `(n, dim)` array, such as OpenAI or Cohere embeddings.

## Command line

```
laya-toolselect select TOOLS.json "request" [-k 5|auto] [--format names|openai|anthropic|mcp]
                [--examples learned.json] [--no-semantic] [--judge english|multilingual]
                [--device cpu|cuda|mps] [--explain]
```

## Production checklist

- Install `[semantic]`. Hybrid beat keyword-only by 12.8 top-5 points on MetaTool and tied it on BFCL.
- Build one selector at startup and call `index()` once.
- Use `k="auto"`, or `k=5` when you need a predictable prompt size.
- Always expose `search_tool()` so a missed tool can still be found.
- Call `learn()` after successful calls, and persist with `save_examples` / `load_examples`.
- Put tools that must always be available (for example `ask_user`) in `always_include`.
- Write tool names and descriptions using the words your users actually say. Avoid near-duplicate tools.
- For non-English traffic, use a multilingual embedder and the multilingual Laya judge. Measure first.
- Before relying on the Laya judge, run `benchmarks/run_laya.py` on your hardware.

## Publishing a release to PyPI

```bash
python -m build
twine check dist/*
twine upload -r testpypi dist/*     # try first
twine upload dist/*
```
