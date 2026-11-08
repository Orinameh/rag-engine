from abc import ABC, abstractmethod

import numpy as np
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ENC = tiktoken.get_encoding("cl100k_base")


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, blocks: list[dict]) -> list[dict]: ...


class TokenChunker(BaseChunker):
    def __init__(self, size: int = 800, overlap: int = 120):
        self.size, self.overlap = size, overlap

    def chunk(self, blocks: list[dict]) -> list[dict]:
        out = []
        for b in blocks:
            ids = _ENC.encode(b["text"])
            step = self.size - self.overlap
            for i, start in enumerate(range(0, len(ids), step)):
                text = _ENC.decode(ids[start : start + self.size])
                if text.strip():
                    out.append(
                        {"text": text, "metadata": {**b["metadata"], "chunk_index": i}}
                    )
        return out


class RecursiveChunker(BaseChunker):
    def __init__(self, size: int = 800, overlap: int = 120):
        self.splitter = RecursiveCharacterTextSplitter(
            # Sizes are in tiktoken tokens (see token_count on Chunk), not
            # characters — without a custom length_function the splitter
            # defaults to characters and produces ~4x smaller chunks.
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=lambda text: len(_ENC.encode(text)),
            separators=[
                "\n# ",
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n##### ",
                "\n###### ",
                "\n\n",
                "\n",
                ". ",
                " ",
            ],
        )

    def chunk(self, blocks: list[dict]) -> list[dict]:
        out = []
        for b in blocks:
            for i, text in enumerate(self.splitter.split_text(b["text"])):
                if text.strip():
                    out.append(
                        {"text": text, "metadata": {**b["metadata"], "chunk_index": i}}
                    )
        return out


class SemanticChunker(BaseChunker):
    def __init__(self, model_name: str = "BAAI/bge-m3", threshold: float = 0.72):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def chunk(self, blocks: list[dict]) -> list[dict]:
        out = []
        for b in blocks:
            sentences = [
                s.strip() for s in b["text"].replace("\n", " ").split(". ") if s.strip()
            ]
            if len(sentences) <= 1:
                out.append(
                    {"text": b["text"], "metadata": {**b["metadata"], "chunk_index": 0}}
                )
                continue
            vecs = self.model.encode(sentences, normalize_embeddings=True)
            sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
            groups, current = [], [sentences[0]]
            for sent, sim in zip(sentences[1:], sims, strict=True):
                if sim < self.threshold:
                    groups.append(". ".join(current) + ".")
                    current = [sent]
                else:
                    current.append(sent)
            groups.append(". ".join(current) + ".")
            for i, text in enumerate(groups):
                out.append(
                    {"text": text, "metadata": {**b["metadata"], "chunk_index": i}}
                )
        return out
