from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter
from unstructured.partition.auto import partition


def parse_document(path: str) -> list[dict]:
    ext = Path(path).suffix.lower()
    if ext == ".md":
        return _parse_markdown(path)
    elements = partition(filename=path, strategy="hi_res")
    return [
        {
            "text": el.text.strip(),
            "metadata": {
                "source": path,
                "page": getattr(el.metadata, "page_number", None),
                "category": el.category,
            },
        }
        for el in elements
        if el.text and el.text.strip()
    ]


def _parse_markdown(path: str) -> list[dict]:
    # All six ATX heading levels. "#####" and deeper are common in runbooks,
    # product docs, and wikis; splitting below them keeps deeply nested
    # sections chunkable instead of dumping a whole page into one block.
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ],
        strip_headers=False,
    )
    text = Path(path).read_text(encoding="utf-8")
    return [
        {
            "text": d.page_content,
            "metadata": {**d.metadata, "source": path, "page": None},
        }
        for d in splitter.split_text(text)
    ]
