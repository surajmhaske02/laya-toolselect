"""Real-component tests: bundled WordLlama embeddings (offline) and the CLI."""
import json
import subprocess
import sys

import pytest

from laya_toolselect import ToolSelector, WordLlamaEmbedder

wordllama = pytest.importorskip("wordllama")

TOOLS = [
    {"name": "get_weather", "description": "Current weather and forecast for a city"},
    {"name": "book_table", "description": "Reserve a table at a restaurant"},
    {"name": "convert_currency", "description": "Convert an amount of money between currencies"},
    {"name": "send_sms", "description": "Send a text message to a phone number"},
]


def test_wordllama_offline_semantics():
    emb = WordLlamaEmbedder()
    v = emb(["will it rain tomorrow", "Current weather and forecast for a city", "Reserve a table"])
    assert v.shape == (3, 256)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_semantic_finds_paraphrase_without_word_overlap():
    s = ToolSelector(TOOLS, embedder="wordllama", alpha=0.0)
    assert s.select("will it rain tomorrow?", k=1).names == ["get_weather"]
    assert s.select("how many euros is 100 dollars", k=1).names == ["convert_currency"]


def test_cli(tmp_path):
    f = tmp_path / "tools.json"
    f.write_text(json.dumps(TOOLS))
    out = subprocess.run([sys.executable, "-m", "laya_toolselect", "select", str(f),
                          "dinner reservation for two", "-k", "1", "--explain"],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == ["book_table"]
    assert "schema text saved" in out.stderr
