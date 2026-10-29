"""OpenAI function calling with a large tool catalog: send only the selected tools.

pip install "laya-toolselect[semantic]" openai
"""
import json

from openai import OpenAI

from laya_toolselect import ToolSelector

ALL_TOOLS = json.load(open("my_tools.json"))      # hundreds of {"type": "function", "function": {...}}
selector = ToolSelector(ALL_TOOLS)                 # index once at startup
client = OpenAI()


def ask(user_message: str):
    picked = selector.select(user_message, k="auto")         # usually 2-6 tools
    tools = picked.tools + [selector.search_tool("openai")]  # escape hatch: the model can search for more
    messages = [{"role": "user", "content": user_message}]
    while True:
        resp = client.chat.completions.create(model="gpt-4.1-mini", messages=messages, tools=tools)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            if call.function.name == "search_tools":
                found = selector.handle_search(call.function.arguments, fmt="openai")
                tools += [t for t in found if t not in tools]   # expose them next turn
                result = [t["function"]["name"] for t in found]
            else:
                result = run_my_tool(call.function.name, json.loads(call.function.arguments))
                selector.learn(user_message, call.function.name)  # optional: learn from usage
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})


def run_my_tool(name, args):
    raise NotImplementedError(name)
