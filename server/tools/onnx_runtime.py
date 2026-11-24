"""Run an exported model with onnxruntime + tokenizers alone.

This is the production inference path: it imports no torch and no
transformers, which is the whole reason it fits on a small Render instance.
Models come from tools/onnx_export.py. It moves into ingestion/ and
vectorstore/ when the embedder (Phase 2) and reranker (Phase 5) adopt it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class OnnxModel:
    def __init__(self, directory: Path, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(self.meta["max_length"])
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads      # Render's shared CPUs
        self.session = ort.InferenceSession(
            str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def _feed(self, encodings) -> dict:
        arrays = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        # The graph's own input list, not the tokenizer's: an exporter prunes
        # inputs the model ignores (DeBERTa never reads token_type_ids).
        return {i.name: arrays[i.name] for i in self.session.get_inputs()}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled, L2-normalised sentence vectors (sentence-transformers' recipe)."""
        encodings = self.tokenizer.encode_batch(texts)
        feed = self._feed(encodings)
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)

    def score(self, queries: list[str], passages: list[str]) -> np.ndarray:
        """Cross-encoder logits for each (query, passage) pair."""
        encodings = self.tokenizer.encode_batch(list(zip(queries, passages)))
        return self.session.run(None, self._feed(encodings))[0]
