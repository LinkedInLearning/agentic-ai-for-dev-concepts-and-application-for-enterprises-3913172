"""Drop-in replacement for the `langchain_pymupdf4llm` package.

The upstream package pulls in the PyMuPDF/pymupdf4llm stack, which is
AGPL-3.0 licensed (or requires a commercial license) as of version 1.28.0.
This local module provides the same import path, class name, and
constructor/.load() signature used by the course notebooks, but is backed
by `pypdf` (BSD-3-Clause) instead, so no AGPL dependency is pulled into
the published repo.

Because this file sits next to the notebooks, Python resolves
`import langchain_pymupdf4llm` to this module before it looks at any
pip-installed package of the same name, so the notebooks require no
code changes.
"""

from pathlib import Path
from typing import Iterator, Literal

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader

__all__ = ["PyMuPDF4LLMLoader"]


class PyMuPDF4LLMLoader(BaseLoader):
    def __init__(self, file_path: str, mode: Literal["page", "single"] = "page", **kwargs):
        self.file_path = str(file_path)
        self.mode = mode

    def lazy_load(self) -> Iterator[Document]:
        reader = PdfReader(self.file_path)
        source = Path(self.file_path).name

        if self.mode == "single":
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            yield Document(page_content=text, metadata={"source": source, "total_pages": len(reader.pages)})
            return

        for i, page in enumerate(reader.pages):
            yield Document(
                page_content=page.extract_text() or "",
                metadata={"source": source, "page": i, "total_pages": len(reader.pages)},
            )

    def load(self):
        return list(self.lazy_load())
