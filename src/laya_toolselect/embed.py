"""Semantic embedders. Each is a callable ``texts -> np.ndarray (n, dim)``.

* ``WordLlamaEmbedder`` -- default. 16 MB static embeddings shipped inside the ``wordllama``
  pip package; no download, no GPU, ~microseconds per text. Install ``laya-toolselect[semantic]``.
* ``SentenceTransformerEmbedder`` -- any sentence-transformers model (stronger, slower).
* ``CallableEmbedder`` -- wrap your own function (OpenAI embeddings, Cohere, a vector DB, ...).
"""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import threading
from typing import Callable, Optional, Sequence

import numpy as np


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class CallableEmbedder:
    def __init__(self, fn: Callable[[Sequence[str]], object], name: str = "custom"):
        self.fn, self.name = fn, name

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return _normalize(np.asarray(self.fn(list(texts))))


class WordLlamaEmbedder:
    """Offline static embeddings from the ``wordllama`` package's bundled weights.

    wordllama 0.4 looks for its bundled tokenizer under ``tokenizer/`` but ships it under
    ``tokenizers/``, and then tries Hugging Face. We point it at a private cache that holds
    the bundled file, so loading never touches the network.
    """
    _lock = threading.Lock()

    def __init__(self, config: str = "l2_supercat", dim: int = 256):
        self.name = f"wordllama-{config}-{dim}"
        self.config, self.dim = config, dim
        self._model = None

    def _load(self):
        try:
            import wordllama
            from wordllama import WordLlama
        except ImportError as e:
            raise ImportError("semantic retrieval needs: pip install 'laya-toolselect[semantic]'") from e
        pkg = Path(wordllama.__file__).parent
        cache = Path(tempfile.gettempdir()) / "laya_toolselect_wordllama"
        for sub in ("tokenizer", "tokenizers"):
            (cache / sub).mkdir(parents=True, exist_ok=True)
            for f in (pkg / "tokenizers").glob("*.json"):
                dst = cache / sub / f.name
                if not dst.exists():
                    shutil.copyfile(f, dst)
        try:
            return WordLlama.load(config=self.config, dim=self.dim, cache_dir=cache, disable_download=True)
        except Exception:
            return WordLlama.load(config=self.config, dim=self.dim)  # fall back to the library's own logic

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(self._model.embed(list(texts), norm=True))


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None,
                 query_prefix: str = "", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.name = f"st-{model}"
        self.model = SentenceTransformer(model, device=device)
        self.query_prefix, self.batch_size = query_prefix, batch_size

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return _normalize(self.model.encode(list(texts), batch_size=self.batch_size,
                                            normalize_embeddings=True, show_progress_bar=False))


def resolve_embedder(embedder) -> Optional[Callable]:
    """``"auto"`` -> WordLlama if installed else None; ``"wordllama"``; ``None``; or any callable."""
    if embedder is None or embedder is False:
        return None
    if embedder == "auto":
        try:
            import wordllama  # noqa: F401
            return WordLlamaEmbedder()
        except ImportError:
            return None
    if embedder == "wordllama":
        return WordLlamaEmbedder()
    if isinstance(embedder, str):
        return SentenceTransformerEmbedder(embedder)
    if callable(embedder):
        return embedder if hasattr(embedder, "name") else CallableEmbedder(embedder)
    raise TypeError("embedder must be 'auto', 'wordllama', a sentence-transformers model id, a callable, or None")
