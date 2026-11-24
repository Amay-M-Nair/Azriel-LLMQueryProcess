"""Export the local models to ONNX, once, on your own machine.

    python -m tools.onnx_export            # from server/: all models
    python -m tools.onnx_export embedder   # just one

Production (Render) runs these with onnxruntime + tokenizers alone: no torch,
no transformers, a fraction of the memory. PyTorch is needed only here, to
do the export, and for offline ingestion.

Each export is checked against the PyTorch model it came from before it is
kept, because an ONNX graph that quietly disagrees with the original would
make every retrieval number after the swap meaningless.

Output goes to data/models/<name>/ (git-ignored): model.onnx + tokenizer.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import config

MODELS_DIR = config.DATA_DIR / "models"


@dataclass(frozen=True)
class Spec:
    name: str
    repo: str
    kind: str              # "embedder" (mean-pooled hidden state) | "cross" (logits)
    max_length: int


SPECS = {
    "embedder": Spec("embedder", config.EMBED_MODEL, "embedder", 256),
    "reranker": Spec("reranker", "cross-encoder/ms-marco-MiniLM-L6-v2", "cross", 512),
    "nli": Spec("nli", "cross-encoder/nli-deberta-v3-base", "cross", 512),
}


def _export(spec: Spec, out: Path) -> None:
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.repo)
    loader = AutoModel if spec.kind == "embedder" else AutoModelForSequenceClassification
    model = loader.from_pretrained(spec.repo).eval()

    sample = tokenizer(["a sample sentence", "and another"], ["a pair", "of texts"]
                       if spec.kind == "cross" else None,
                       padding=True, return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    output = "last_hidden_state" if spec.kind == "embedder" else "logits"

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args):
            result = self.inner(**dict(zip(names, args)))
            return result.last_hidden_state if spec.kind == "embedder" else result.logits

    axes = {n: {0: "batch", 1: "sequence"} for n in names}
    axes[output] = {0: "batch", 1: "sequence"} if spec.kind == "embedder" else {0: "batch"}

    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            Wrapped(model), tuple(sample[n] for n in names), str(out / "model.onnx"),
            input_names=names, output_names=[output], dynamic_axes=axes,
            opset_version=17, dynamo=False,
        )
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    labels = getattr(model.config, "id2label", None)
    (out / "meta.json").write_text(json.dumps({
        "repo": spec.repo, "kind": spec.kind, "max_length": spec.max_length,
        "inputs": names, "labels": labels,
    }, indent=2), encoding="utf-8")


def _check(spec: Spec, out: Path) -> float:
    """Largest disagreement between the ONNX graph and the PyTorch model."""
    import torch
    from tools.onnx_runtime import OnnxModel

    texts = ["Revenue increased 6% year over year.", "The company faces currency risk.",
             "Ignore previous instructions."]
    pairs = ["What was revenue growth?", "Which risks are named?", "Who is the CEO?"]
    onnx_model = OnnxModel(out)

    if spec.kind == "embedder":
        from sentence_transformers import SentenceTransformer
        reference = SentenceTransformer(spec.repo).encode(texts, normalize_embeddings=True)
        got = onnx_model.embed(texts)
        return float(np.abs(reference - got).max())

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec.repo)
    model = AutoModelForSequenceClassification.from_pretrained(spec.repo).eval()
    with torch.no_grad():
        reference = model(**tokenizer(pairs, texts, padding=True, truncation=True,
                                      return_tensors="pt")).logits.numpy()
    got = onnx_model.score(pairs, texts)
    return float(np.abs(reference - got).max())


def export(name: str) -> float:
    spec = SPECS[name]
    out = MODELS_DIR / name
    if out.exists():
        shutil.rmtree(out)
    _export(spec, out)
    error = _check(spec, out)
    if error > 1e-3:
        shutil.rmtree(out)
        raise SystemExit(f"{name}: ONNX disagrees with PyTorch by {error:.2e}; not kept")
    size = (out / "model.onnx").stat().st_size / 1e6
    print(f"{name:9} {spec.repo}  {size:.0f} MB  max |onnx - torch| = {error:.1e}")
    return error


def main(argv: list[str]) -> int:
    for name in argv or list(SPECS):
        export(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
