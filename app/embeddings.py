"""CPU-compatible open-source text embedding wrapper loaded once per worker."""

import math
from typing import Protocol

from fastembed import TextEmbedding


class Embedder(Protocol):
    """Minimal embedding interface allowing deterministic test doubles."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


class FastEmbedder:
    """ONNX Runtime CPU embedder backed by fastembed, initialized once per worker."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        expected_dim: int = 384,
    ) -> None:
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._model = TextEmbedding(model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._expected_dim

    def embed(self, text: str) -> list[float]:
        """Generate a single L2-normalized embedding vector of length `dimension`."""
        raw_vector = next(iter(self._model.embed([text])))
        vec = [float(x) for x in raw_vector]
        if len(vec) != self._expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch for {self._model_name}: "
                f"expected {self._expected_dim}, got {len(vec)}"
            )
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            raise ValueError(f"Zero-norm embedding produced by {self._model_name}")
        if abs(norm - 1.0) > 1e-4:
            vec = [x / norm for x in vec]
        return vec
