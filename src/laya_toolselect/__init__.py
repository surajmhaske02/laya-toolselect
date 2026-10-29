"""laya-toolselect: show an LLM only the tools it needs.

    from laya_toolselect import ToolSelector
    selector = ToolSelector(my_tools)              # OpenAI / Anthropic / MCP / LangChain / callables
    picked = selector.select("book a table for two tonight", k=5)
    client.chat.completions.create(..., tools=picked.tools)
"""
from .embed import CallableEmbedder, SentenceTransformerEmbedder, WordLlamaEmbedder
from .judge import LayaJudge
from .selector import Candidate, Selection, ToolSelector
from .tools import Tool, to_tool, to_tools

__version__ = "0.1.0"
__all__ = ["ToolSelector", "Selection", "Candidate", "Tool", "to_tool", "to_tools", "LayaJudge",
           "WordLlamaEmbedder", "SentenceTransformerEmbedder", "CallableEmbedder", "__version__"]
