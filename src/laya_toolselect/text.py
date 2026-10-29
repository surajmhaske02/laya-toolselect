"""Tokenization for tool text: identifier splitting, light English stemming, CJK bigrams.

Kept dependency-free on purpose. Works on any script: words are runs of non-separator
characters, so Devanagari, Tamil, Arabic etc. stay whole words; Chinese/Japanese/Korean
runs (no spaces) are indexed as character bigrams.
"""
from __future__ import annotations

import re

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEP = re.compile(r"[\s_\-\.:/\\|,;!?\"'`()\[\]{}<>=+*&^%$#@~]+")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

STOPWORDS = frozenset("""
a an the and or of to in on for with by from at as is are was were be been being this that these
those it its i me my we our you your he she they them their what which who whom how can could
would should will shall may might do does did done have has had please want need get some any
all also into about than then there here so if not no yes just like up out over very too via
""".split())

_SUFFIXES = (("ational", "ate"), ("ization", "ize"), ("ations", "ate"), ("ation", "ate"),
             ("ments", "ment"), ("ingly", ""), ("edly", ""), ("ies", "y"), ("ied", "y"),
             ("sses", "ss"), ("ing", ""), ("ers", "er"), ("ed", ""), ("es", "e"), ("s", ""))


def split_identifier(text: str) -> str:
    """``getWeatherForecast`` / ``get_weather.forecast`` -> ``get Weather Forecast``."""
    return _SEP.sub(" ", _CAMEL.sub(" ", str(text))).strip()


def stem(word: str) -> str:
    """Light suffix stripping for ASCII words (``bookings`` -> ``book``, ``flights`` -> ``flight``)."""
    if len(word) <= 3 or not word.isascii() or not word.isalpha():
        return word
    for suf, rep in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            if suf == "s" and word.endswith(("ss", "us", "is")):
                return word
            w = word[: -len(suf)] + rep
            if suf in ("ing", "ed") and len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]                       # running -> run, stopped -> stop
            return w
    return word


def tokenize(text: str, *, use_stem: bool = True, stopwords: frozenset = STOPWORDS) -> list:
    out = []
    for tok in split_identifier(text).lower().split():
        if not tok or tok in stopwords:
            continue
        if _CJK.search(tok):
            if len(tok) == 1:
                out.append(tok)
            else:
                out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
            continue
        out.append(stem(tok) if use_stem else tok)
    return out
