import json

import numpy as np
import pytest

from laya_toolselect import LayaJudge, Tool, ToolSelector, to_tool, to_tools
from laya_toolselect.text import split_identifier, stem, tokenize

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather forecast for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}
FLIGHTS = {"name": "searchFlights", "description": "Search flights between two airports on a date.",
           "input_schema": {"type": "object", "properties": {"origin": {"type": "string"},
                                                             "destination": {"type": "string"}}}}
STOCKS = {"name": "stock_quote", "description": "Latest share price for a ticker symbol.",
          "inputSchema": {"type": "object", "properties": {"ticker": {"type": "string"}}}}
EMAIL = {"type": "function", "name": "send_email", "description": "Send an email message to a recipient.",
         "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}}}


def calendar_add(title: str, when: str, minutes: int = 30):
    """Add an event to the user's calendar."""


ALL = [WEATHER, FLIGHTS, STOCKS, EMAIL, calendar_add]


def lexical(tools=ALL, **kw):
    return ToolSelector(tools, embedder=None, **kw)


class FakeEmbed:
    """Deterministic bag-of-words hashing embedder for tests."""
    name = "fake"

    def __call__(self, texts):
        m = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tokenize(t):
                m[i, hash(w) % 64] += 1
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1
        return m / n


class FakeAgent:
    """Mimics laya Agent.system_one for a choice question; prefers labels in `prefer`."""
    def __init__(self, prefer):
        self.prefer, self.calls = prefer, []

    def system_one(self, state, questions, **kw):
        self.calls.append((state, questions, kw))
        answers = {}
        for qid, q in questions.items():
            labels = list(q["criteria"])
            raw = np.array([5.0 if lab in self.prefer else 1.0 for lab in labels])
            p = raw / raw.sum()
            answers[qid] = {"choice": labels[int(p.argmax())],
                            "probabilities": {lab: float(v) for lab, v in zip(labels, p)}}
        return {"answers": answers, "usage": {"input_tokens": 100}}


# ------------------------------------------------------------------ formats
def test_formats_detected():
    names = [t.name for t in to_tools(ALL)]
    assert names == ["get_weather", "searchFlights", "stock_quote", "send_email", "calendar_add"]
    cal = to_tool(calendar_add)
    assert cal.description == "Add an event to the user's calendar."
    assert cal.parameters["required"] == ["title", "when"]
    assert cal.parameters["properties"]["minutes"]["type"] == "integer"


def test_mcp_object_and_listtools_result():
    class MCPTool:
        name, description = "read_file", "Read a file from disk"
        inputSchema = {"type": "object", "properties": {"path": {"type": "string"}}}

    class ListToolsResult:
        tools = [MCPTool()]
    tools = to_tools(ListToolsResult())
    assert tools[0].name == "read_file" and "path" in tools[0].parameters["properties"]


def test_langchain_like_tool():
    class LCTool:
        name, description = "wiki", "Look up Wikipedia"
        args = {"query": {"type": "string"}}
    assert to_tool(LCTool()).parameters["properties"] == {"query": {"type": "string"}}


def test_bad_tool():
    with pytest.raises(ValueError):
        to_tool({"description": "no name"})


def test_export_roundtrip():
    t = to_tool(FLIGHTS)
    assert t.to_openai()["function"]["name"] == "searchFlights"
    assert "input_schema" in t.to_anthropic() and "inputSchema" in t.to_mcp()


# ------------------------------------------------------------------ text
def test_text_helpers():
    assert split_identifier("getWeatherForecast") == "get Weather Forecast"
    assert split_identifier("stock_price.get") == "stock price get"
    assert stem("flights") == "flight" and stem("booking") == "book" and stem("running") == "run"
    assert stem("status") == "status" and stem("class") == "class"
    assert tokenize("The weather in Paris") == ["weather", "paris"]
    assert "天气" in tokenize("北京天气")                       # CJK bigrams
    assert tokenize("मौसम कैसा है") == ["मौसम", "कैसा", "है"]     # Devanagari words stay whole


# ------------------------------------------------------------------ selection
def test_select_returns_original_objects():
    sel = lexical().select("what's the weather in Pune tomorrow", k=2)
    assert sel.tools[0] is WEATHER
    assert sel.names[0] == "get_weather"
    assert 0 < sel.savings < 1 and sel.total_tools == 5


def test_select_ranks_by_params_and_names():
    s = lexical()
    assert s.select("price of ticker AAPL", k=1).names == ["stock_quote"]
    assert s.select("flights from BOM to DEL", k=1).names == ["searchFlights"]
    assert s.select("add an event to my calendar at 5pm", k=1).names == ["calendar_add"]


def test_k_zero_and_always_include():
    s = lexical()
    assert len(s.select("weather", k=0)) == 0
    sel = s.select("weather", k=1, always_include=["send_email"])
    assert sel.names == ["get_weather", "send_email"]


def test_auto_k_bounds():
    s = lexical()
    sel = s.select("weather", k="auto", min_k=2, max_k=3)
    assert 2 <= len(sel) <= 3


def test_invalid_args():
    s = lexical()
    for bad in [dict(k=-1), dict(k="many"), dict(min_k=0), dict(min_k=4, max_k=2)]:
        with pytest.raises(ValueError):
            s.select("weather", **bad)
    with pytest.raises(ValueError):
        s.select("  ")
    with pytest.raises(ValueError):
        ToolSelector(fusion="max")


def test_empty_selector():
    assert len(ToolSelector(embedder=None).select("anything")) == 0


def test_add_replace_remove():
    s = lexical()
    s.add_tool({"name": "get_weather", "description": "Weather v2 with hourly radar"})
    assert len(s) == 5 and s.select("hourly radar", k=1).names == ["get_weather"]
    s.remove_tools(["get_weather"])
    assert "get_weather" not in [t.name for t in s.tools]
    assert s.select("weather radar", k=1).names != ["get_weather"]


def test_learn_changes_ranking_and_persists(tmp_path):
    s = lexical()
    q = "ping the finance team about the invoice"
    s.learn(q, "send_email")
    assert s.select(q, k=1).names == ["send_email"]
    s.save_examples(tmp_path / "ex.json")
    s2 = lexical()
    s2.load_examples(tmp_path / "ex.json")
    assert s2.select(q, k=1).names == ["send_email"]
    with pytest.raises(KeyError):
        s.learn("x", "nope")


def test_semantic_hybrid_and_rrf():
    for fusion in ["weighted", "rrf"]:
        s = ToolSelector(ALL, embedder=FakeEmbed(), fusion=fusion)
        sel = s.select("weather forecast city", k=3)
        assert sel.names[0] == "get_weather"
        assert sel.candidates[0].semantic > 0


def test_semantic_uses_examples():
    s = ToolSelector(ALL, embedder=FakeEmbed(), alpha=0.0)
    s.learn("is it going to rain", "get_weather")
    assert s.select("is it going to rain", k=1).names == ["get_weather"]


def test_callable_embedder_wrapped():
    s = ToolSelector(ALL, embedder=lambda texts: FakeEmbed()(texts))
    assert s.select("send an email", k=1).names == ["send_email"]


# ------------------------------------------------------------------ judge
def test_judge_one_call_debias_and_fusion():
    agent = FakeAgent(prefer={"send_email"})
    judge = LayaJudge("english", agent=agent)
    s = lexical(judge=judge, shortlist=4, judge_mode="judge")
    sel = s.select("what is the weather", k=2)
    assert len(agent.calls) == 1                      # one forward pass
    state, questions, kw = agent.calls[0]
    assert set(questions) == {"tool", "tool_rev"}      # debias row in the same pass
    crit = list(questions["tool"]["criteria"])
    assert crit[-1] == "none" and len(crit) == 5
    order = questions["tool_rev"]["option_order"]
    assert order[-1] == 4 and sorted(order) == list(range(5))
    assert kw["max_len"] == 512 and kw["head_max_len"] == 512 - 96
    assert sel.names[0] == "send_email"               # judge_mode="judge" trusts Laya
    assert sel.p_none is not None and 0 < sel.p_none < 1
    assert abs(sum(c.judge for c in sel.shortlist[:4]) + sel.p_none - 1) < 1e-6


def test_judge_fuse_mode_keeps_retrieval_signal():
    agent = FakeAgent(prefer={"send_email"})
    s = lexical(judge=LayaJudge("english", agent=agent), shortlist=4, judge_weight=0.5)
    sel = s.select("what is the weather", k=4)
    assert set(sel.names[:2]) == {"get_weather", "send_email"}


def test_judge_auto_k_uses_probabilities():
    agent = FakeAgent(prefer={"get_weather"})
    s = lexical(judge=LayaJudge("english", agent=agent), shortlist=5, judge_mode="judge")
    sel = s.select("weather", k="auto", coverage=0.5, max_k=5)
    assert sel.names == ["get_weather"]


def test_judge_duplicate_and_reserved_labels():
    j = LayaJudge("english", agent=FakeAgent(set()), debias=False)
    res = j.score("q", [Tool("a", "x"), Tool("a", "y"), Tool("none", "z")])
    assert len(res["probs"]) == 3
    labels = list(j.agent.calls[0][1]["tool"]["criteria"])
    assert labels == ["a", "a#2", "tool_none", "none"]


def test_judge_validation():
    with pytest.raises(ValueError):
        LayaJudge("klingon")
    with pytest.raises(ValueError):
        LayaJudge("english", query_tokens=500)


# ------------------------------------------------------------------ meta tool
def test_search_tool_roundtrip():
    s = lexical()
    spec = s.search_tool("openai")
    assert spec["function"]["name"] == "search_tools"
    found = s.handle_search(json.dumps({"query": "share price", "k": 2}), fmt="anthropic")
    assert found[0]["name"] == "stock_quote" and "input_schema" in found[0]
