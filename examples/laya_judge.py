"""Add the Laya judge: one local forward pass re-scores the shortlist and says when no tool fits.

pip install "laya-toolselect[all]"   # first use downloads the Laya checkpoint from Hugging Face
"""
from laya_toolselect import LayaJudge, ToolSelector

tools = [
    {"name": "get_weather", "description": "Current weather and forecast for a city"},
    {"name": "book_table", "description": "Reserve a table at a restaurant"},
    {"name": "convert_currency", "description": "Convert an amount between currencies"},
]
judge = LayaJudge("multilingual", device=None)      # 100+ languages; "english" also available
selector = ToolSelector(tools, judge=judge)

for request in ["Will it rain in Pune tomorrow?",
                "कल रात दो लोगों के लिए टेबल बुक करो",          # Hindi: book a table for two tomorrow night
                "Tell me a joke"]:
    res = selector.select(request, k="auto")
    print(request, "->", res.names, f"p(no tool)={res.p_none:.2f}")
    if res.p_none > 0.6:
        print("  answer directly, no tools needed")
