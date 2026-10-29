# laya-toolselect v0.1.0 benchmark report

## Question

Given a catalog of many tools and one user request, does the correct tool end up among the
few tools we show the LLM? If it doesn't, the LLM cannot call it. If we show too many,
prompt cost and confusion rise (RAG-MCP: 13.6% correct tool choice with all tools in the
prompt vs 43.1% with retrieval first).

## Data (public, from GitHub)

| | MetaTool | BFCL pooled |
|---|---|---|
| Source | HowieHwong/MetaTool `dataset/` | ShishirPatil/gorilla `bfcl_eval/data` (v4) |
| Catalog | 199 real ChatGPT-plugin tools (name + description) | 1,096 functions with JSON schemas, pooled from simple_python, multiple, live_simple and live_multiple |
| Requests | 20,614 labeled (request, tool) pairs. Per tool: half kept as a "usage log", half for evaluation | 1,911 single-call requests |
| Dev / test | 500 / 2,000 | 573 / 1,338 |
| Difficulty | Many near-duplicate tools (`ResearchHelper` / `ResearchFinder`, several finance tools) | Large catalog, many APIs share vocabulary |

`benchmarks/prepare.py` rebuilds both files (seed 13). For BFCL, functions are keyed by name and
4,297 repeated definitions of the same names were dropped. All settings (blend weight 0.3, auto
margin 0.3, no parameter text in embeddings) were chosen on **dev**. Test was run once.

## Results (test split)

R@k = gold tool among the top k. "auto" = `k="auto"`: tools exposed, how often the gold tool is
among them, and the share of the full catalog's tool-definition characters sent. CPU latency
per request on a 2-vCPU cloud VM, one Python thread, index already built.

| dataset | method | R@1 | R@3 | R@5 | R@10 | MRR@10 | auto: tools | auto: recall | auto: schema sent | ms/req |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaTool | keyword, description only (naive baseline) | 0.359 | 0.470 | 0.529 | 0.587 | 0.429 | 3.1 | 0.478 | 1.5% | 0.10 |
| MetaTool | bm25 (names + params + stemming) | 0.435 | 0.578 | 0.626 | 0.693 | 0.518 | 2.6 | 0.569 | 1.4% | 0.07 |
| MetaTool | semantic (WordLlama) | 0.495 | 0.679 | 0.735 | 0.808 | 0.600 | 5.0 | 0.760 | 2.8% | 0.34 |
| MetaTool | **hybrid (default)** | **0.514** | **0.694** | **0.754** | **0.825** | **0.617** | 4.0 | 0.750 | 2.2% | 0.36 |
| MetaTool | hybrid + learn 5/tool | 0.576 | 0.765 | 0.831 | 0.887 | 0.683 | 3.8 | 0.829 | 2.0% | 0.56 |
| MetaTool | hybrid + learn 20/tool | 0.672 | 0.834 | 0.884 | 0.926 | 0.762 | 3.6 | 0.879 | 1.9% | 0.83 |
| MetaTool | **hybrid + learn 50/tool** | **0.730** | **0.880** | **0.914** | **0.948** | **0.811** | 3.4 | **0.918** | 1.9% | 0.94 |
| BFCL | keyword, description only (naive baseline) | 0.464 | 0.613 | 0.680 | 0.750 | 0.557 | 3.3 | 0.645 | 0.3% | 0.21 |
| BFCL | bm25 (names + params + stemming) | 0.641 | 0.821 | 0.879 | 0.926 | 0.742 | 3.0 | 0.857 | 0.3% | 0.35 |
| BFCL | semantic (WordLlama) | 0.520 | 0.708 | 0.782 | 0.875 | 0.633 | 6.6 | 0.847 | 0.6% | 0.74 |
| BFCL | **hybrid (default)** | **0.652** | **0.828** | 0.870 | **0.948** | **0.753** | 5.2 | **0.919** | 0.5% | 0.72 |

Paired bootstrap 95% intervals on R@5 (5,000 resamples):

- Hybrid vs naive baseline: MetaTool +0.225 [+0.204, +0.244]; BFCL +0.190 [+0.166, +0.214]
- Hybrid vs bm25: MetaTool +0.128 [+0.112, +0.143]; BFCL −0.009 [−0.023, +0.005] (tie)
- Hybrid vs semantic: MetaTool +0.019 [+0.006, +0.033]; BFCL +0.088 [+0.071, +0.105]
- Learning 50 past requests per tool: MetaTool +0.160 [+0.141, +0.178]

Other variants (more results on the full table in `results/test.json`): keyword-only learning
reaches about the same R@5 (0.906) as hybrid learning (0.914) once 50 examples per tool exist.

## Findings

1. **Hybrid is the most robust default.** Keywords win on BFCL, where requests reuse API
   vocabulary. Embeddings win on MetaTool, where requests are paraphrases. The hybrid is best or
   tied on both, and at R@10 it is best on both.
2. **Learning from usage is the biggest lever** (+16 R@5 points with 50 requests per tool, +8 with
   only 5). Calling `selector.learn(request, tool_used)` after successful tool calls compounds.
3. **`k="auto"` adapts the list length to the request.** On BFCL it sends 5.2 tools on average
   with 0.919 recall, versus 0.870 at a fixed k=5.
4. **Cost is negligible.** Under 1 ms per request and under 1.2 s to index 1,096 tools. The
   bundled embeddings need no network and no GPU.

## Not measured here

- **The Laya judge's quality and speed.** This environment cannot download model weights from
  Hugging Face. The judge's wiring was checked against the released `laya` 0.4.17 source
  (`tests/test_laya_contract.py`). Every question passes Laya's own validator and option
  renderer, the reversed second question puts the options in the intended slots, averaging
  the two rows cancels a pure position bias exactly, and probabilities are read by label after
  Laya's own un-permuting. That checks wiring, not accuracy. `benchmarks/run_laya.py` measures
  re-ranking gains, latency and no-tool detection (AUROC of `p_none` on 240 BFCL irrelevance
  vs 400 relevant requests) on a machine that can download the checkpoint.
- End-to-end LLM task success. These are retrieval metrics: an LLM can still pick wrongly among
  correct candidates.
- Stronger embedders (bge, e5, OpenAI). They plug in via `embedder=`. Expected to beat WordLlama
  on semantic recall at a higher cost.
- Non-English requests on these catalogs.
