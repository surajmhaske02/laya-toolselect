"""Check LayaJudge against the installed laya package's own validator and option renderer.

Laya's question validation, option rendering and probability un-permuting are pure Python,
but laya's modules import torch at the top. These functions are read from the installed
source with `ast` and run in isolation, so the test needs `laya` installed (even with
`pip install --no-deps laya`) but not torch or model weights. It checks wiring, not quality.
"""
import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from laya_toolselect import LayaJudge, Tool, ToolSelector

spec = importlib.util.find_spec("laya")
if spec is None or not spec.submodule_search_locations:
    pytest.skip("laya not installed", allow_module_level=True)
LAYA = Path(list(spec.submodule_search_locations)[0])


def _extract(path, names, ns, cls=None):
    body = ast.parse(path.read_text()).body
    if cls:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    nodes = [n for n in body if (isinstance(n, ast.FunctionDef) and n.name in names)
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    for n in nodes:
        n.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    missing = names - set(ns)
    assert not missing, f"laya source changed, missing {missing}"


NS = {"Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "json": json, "np": np}
_extract(LAYA / "common.py", {"QTYPES", "render_criterion", "_resolve_noul_labels", "render_options",
                              "unpermute_probs"}, NS)
_extract(LAYA / "agent.py", {"_option_count"}, NS)
_extract(LAYA / "agent.py", {"_check_question", "_to_internal"}, NS, cls="Agent")


class ContractAgent:
    """Validates with laya's code; answers like Agent._decode_answers with a slot-0 bias."""
    def system_one(self, state, questions, max_len=None, head_max_len=None):
        assert max_len and head_max_len and head_max_len < max_len
        answers = {}
        for qid, q in questions.items():
            NS["_check_question"](qid, q)
            internal = NS["_to_internal"](q)
            opts = NS["render_options"](internal)
            keys = list(internal["crit"])
            order = internal.get("option_order") or list(range(len(keys)))
            slot_p = np.linspace(2, 1, len(keys))
            slot_p /= slot_p.sum()
            p = NS["unpermute_probs"](slot_p, internal.get("option_order"))
            answers[qid] = {"choice": keys[int(p.argmax())],
                            "probabilities": {k: float(v) for k, v in zip(keys, p)},
                            "_slots": [opts[i] for i in order]}
        self.last = answers
        return {"answers": answers, "usage": {}}


@pytest.mark.parametrize("model", ["english", "multilingual"])
def test_questions_valid_and_debias_cancels_position_bias(model):
    agent = ContractAgent()
    tools = [Tool(f"tool_{i}", f"Description {i} " * 20) for i in range(10)] + [Tool("none", "clash")]
    res = LayaJudge(model, agent=agent).score("book a cab", tools)
    rows = agent.last
    assert rows["tool"]["_slots"][0].startswith("tool_0:")
    assert rows["tool_rev"]["_slots"][0].startswith("tool_none:")
    assert all(r["_slots"][-1].startswith("none:") for r in rows.values())
    assert max(res["probs"]) - min(res["probs"]) < 1e-9          # slot bias averaged out
    assert abs(sum(res["probs"]) + res["p_none"] - 1) < 1e-9


def test_selector_with_contract_agent_and_single_tool():
    sel = ToolSelector([{"name": "only", "description": "x"}], embedder=None,
                       judge=LayaJudge("english", agent=ContractAgent()))
    r = sel.select("q", k=1)
    assert r.names == ["only"] and 0 < r.p_none < 1
