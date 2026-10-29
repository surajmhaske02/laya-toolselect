# Changelog

## 0.1.0 (2026-10-29)

First release.

- `ToolSelector`: hybrid keyword + semantic tool retrieval, `k="auto"`, `always_include`.
- Tool formats: OpenAI (Chat and Responses), Anthropic, MCP, LangChain, Python callables.
- `learn()` / `save_examples()` / `load_examples()` to learn from usage.
- `search_tool()` / `handle_search()` meta-tool fallback.
- Optional `LayaJudge` (one forward pass, position-debiased, `p_none`).
- CLI `laya-toolselect select`.
- Benchmarks on MetaTool (199 tools) and pooled BFCL v4 (1,096 tools).
