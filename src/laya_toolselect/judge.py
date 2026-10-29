"""Laya as the tool judge: one forward pass scores a whole shortlist.

The shortlist becomes the options of a Laya ``choice`` question, plus a ``none`` option
("no listed tool fits"). Laya returns a probability for every option, so a single pass
gives a ranking of the shortlist *and* a no-tool probability. ``debias=True`` asks the
same question a second time with the options reversed (a second row in the same forward
pass) and averages, which cancels position preference.

Laya is optional: install ``laya-toolselect[laya]``. Any object with
``system_one(state, questions, max_len=, head_max_len=)`` (e.g. ``laya.load(...)`` or a fake
for tests) can be passed as ``agent``.
"""
from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

from .tools import Tool

CHECKPOINTS = {
    "english": ("convaiinnovations/laya", None, 512),
    "multilingual": ("convaiinnovations/laya", "multilingual", 1024),
}
NONE_LABEL = "none"
DEFAULT_INSTRUCTIONS = ("Which tool should the assistant call to complete the user's request? "
                        "Choose none if no listed tool can do it.")
NONE_DESCRIPTION = "no listed tool fits; answer directly or ask the user"


class LayaJudge:
    def __init__(self, model: str = "multilingual", *, device: Optional[str] = None,
                 revision: Optional[str] = None, agent: Any = None, max_len: Optional[int] = None,
                 query_tokens: int = 96, debias: bool = True,
                 instructions: str = DEFAULT_INSTRUCTIONS, option_chars: int = 220):
        if model not in CHECKPOINTS and agent is None and "/" not in model:
            raise ValueError(f"model must be one of {sorted(CHECKPOINTS)} or a checkpoint path")
        repo, sub, default_len = CHECKPOINTS.get(model, (model, None, 512))
        self.model, self.repo, self.subfolder = model, repo, sub
        self.max_len = max_len or default_len
        if not 32 <= query_tokens < self.max_len - 64:
            raise ValueError("query_tokens must leave room for the options")
        self.head_max_len = self.max_len - query_tokens
        self.device, self.revision, self.debias = device, revision, debias
        self.instructions, self.option_chars = instructions, option_chars
        self._agent, self._lock = agent, threading.Lock()

    @property
    def agent(self):
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    try:
                        import laya
                    except ImportError as e:
                        raise ImportError("LayaJudge needs: pip install 'laya-toolselect[laya]'") from e
                    self._agent = laya.load(self.repo, subfolder=self.subfolder, device=self.device,
                                            revision=self.revision)
        return self._agent

    def _labels(self, tools: Sequence[Tool]) -> list:
        seen, labels = {}, []
        for t in tools:
            base = t.name if t.name != NONE_LABEL else "tool_none"
            n = seen.get(base, 0)
            seen[base] = n + 1
            labels.append(base if n == 0 else f"{base}#{n + 1}")
        return labels

    def score(self, query: str, tools: Sequence[Tool], context: Optional[str] = None) -> dict:
        """Return ``{"probs": [p per tool, input order], "p_none": float, "usage": dict}``."""
        if not tools:
            return {"probs": [], "p_none": 1.0, "usage": {}}
        labels = self._labels(tools)
        # Laya renders each option as "label: description" and caps it at 48 tokens; when the
        # options overflow head_max_len it cuts every option to (head_max_len - 16) // n tokens
        # and squeezes the instruction. Size descriptions (~4 chars/token) so all options fit
        # with ~48 tokens left for the instruction.
        n_opts = len(labels) + 1
        per_tok = max(8, min(48, (self.head_max_len - 48) // n_opts - 1))
        criteria = {}
        for lab, t in zip(labels, tools):
            room = min(self.option_chars, per_tok * 4) - len(lab) - 2
            criteria[lab] = t.description[: max(0, room)].strip() or None
        criteria[NONE_LABEL] = NONE_DESCRIPTION
        q = {"type": "choice", "instructions": self.instructions, "criteria": criteria}
        questions = {"tool": q}
        if self.debias and len(criteria) > 2:
            n = len(criteria)
            # reverse the tools, keep "none" last in both rows
            questions["tool_rev"] = dict(q, option_order=list(range(n - 2, -1, -1)) + [n - 1])
        state = {"request": query} if context is None else {"conversation": context, "request": query}
        out = self.agent.system_one(state, questions, max_len=self.max_len, head_max_len=self.head_max_len)
        rows = [out["answers"][k]["probabilities"] for k in questions]
        avg = {lab: sum(r[lab] for r in rows) / len(rows) for lab in criteria}
        return {"probs": [float(avg[lab]) for lab in labels], "p_none": float(avg[NONE_LABEL]),
                "usage": out.get("usage", {})}
