"""One internal tool shape, built from whatever format the developer already has.

Supported inputs (auto-detected):

* OpenAI Chat Completions: ``{"type": "function", "function": {"name", "description", "parameters"}}``
* OpenAI Responses API:    ``{"type": "function", "name", "description", "parameters"}``
* Anthropic Messages API:  ``{"name", "description", "input_schema"}``
* MCP ``tools/list``:      ``{"name", "description", "inputSchema"}`` or an ``mcp.types.Tool`` object
* LangChain ``BaseTool`` (``name``, ``description``, ``args``)
* Plain Python callables (name from ``__name__``, description from the docstring)
* :class:`Tool` itself

The original object is kept on ``Tool.raw``, so a selection hands back exactly what was passed in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
from typing import Any, Iterable, Optional

from .text import split_identifier


@dataclass
class Tool:
    name: str
    description: str = ""
    parameters: dict = field(default_factory=dict)
    examples: list = field(default_factory=list)   # example user requests this tool handles
    tags: list = field(default_factory=list)
    raw: Any = None

    # ---------------------------------------------------------------- rendering
    def param_text(self) -> str:
        """Parameter names and descriptions, flattened one level of nesting."""
        props = (self.parameters or {}).get("properties") or {}
        parts = []
        for pname, spec in props.items():
            parts.append(split_identifier(str(pname)))
            if isinstance(spec, dict):
                if spec.get("description"):
                    parts.append(str(spec["description"]))
                if isinstance(spec.get("enum"), list):
                    parts.append(" ".join(str(v) for v in spec["enum"][:20]))
                sub = spec.get("properties") or (spec.get("items") or {}).get("properties") or {}
                for sname, sspec in list(sub.items())[:20]:
                    parts.append(split_identifier(str(sname)))
                    if isinstance(sspec, dict) and sspec.get("description"):
                        parts.append(str(sspec["description"]))
        return " ".join(parts)

    def card(self, max_chars: int = 400) -> str:
        """Short human-readable card: what the semantic embedder and Laya read."""
        text = f"{split_identifier(self.name)}: {self.description}".strip()
        return text[:max_chars]

    def schema_chars(self) -> int:
        """Size of the tool definition as an LLM would receive it (JSON characters)."""
        return len(json.dumps({"name": self.name, "description": self.description,
                               "parameters": self.parameters}, ensure_ascii=False))

    # ---------------------------------------------------------------- export
    def to_openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters or {"type": "object", "properties": {}}}}

    def to_anthropic(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.parameters or {"type": "object", "properties": {}}}

    def to_mcp(self) -> dict:
        return {"name": self.name, "description": self.description,
                "inputSchema": self.parameters or {"type": "object", "properties": {}}}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _schema_of_callable(fn) -> dict:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}}
    types = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}
    props, required = {}, []
    for pname, p in sig.parameters.items():
        if pname in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[pname] = {"type": types.get(p.annotation, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": props, "required": required}


def to_tool(obj: Any) -> Tool:
    """Convert one tool definition in any supported format to :class:`Tool`."""
    if isinstance(obj, Tool):
        return obj
    raw = obj
    # OpenAI chat format wraps the function.
    fn = _get(obj, "function")
    if isinstance(fn, dict) and _get(obj, "type", "function") == "function":
        obj = fn
    name = _get(obj, "name")
    if name is None and callable(obj):
        doc = inspect.getdoc(obj) or ""
        return Tool(name=obj.__name__, description=doc.strip(), parameters=_schema_of_callable(obj), raw=raw)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"cannot read a tool name from {type(raw).__name__}: {str(raw)[:120]}")
    desc = _get(obj, "description") or ""
    params = (_get(obj, "parameters") or _get(obj, "input_schema") or _get(obj, "inputSchema")
              or _get(obj, "args_schema") or {})
    if not isinstance(params, dict):
        # pydantic model class (LangChain args_schema) or object with model_json_schema
        for attr in ("model_json_schema", "schema"):
            f = getattr(params, attr, None)
            if callable(f):
                try:
                    params = f()
                    break
                except Exception:
                    pass
        else:
            params = {}
        if not isinstance(params, dict):
            params = {}
    if not params and isinstance(_get(obj, "args"), dict):       # LangChain BaseTool.args
        params = {"type": "object", "properties": _get(obj, "args")}
    examples = list(_get(obj, "examples", None) or [])
    tags = list(_get(obj, "tags", None) or [])
    return Tool(name=name.strip(), description=str(desc).strip(), parameters=params,
                examples=[str(e) for e in examples], tags=[str(t) for t in tags], raw=raw)


def to_tools(objs: Iterable[Any]) -> list:
    """Convert a list of tool definitions; MCP ``ListToolsResult`` objects are unwrapped."""
    tools_attr = getattr(objs, "tools", None)
    if tools_attr is not None and not isinstance(objs, (list, tuple)):
        objs = tools_attr
    return [to_tool(o) for o in objs]
