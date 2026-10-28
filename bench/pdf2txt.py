"""PDF -> plain text (the PDF modality): PyMuPDF text extraction in reading order."""
import sys
from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_text(path: Path) -> str:
    with fitz.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


if __name__ == "__main__":
    for pdf in sorted(Path(sys.argv[1]).glob("*.pdf")):
        text = pdf_to_text(pdf)
        pdf.with_suffix(".txt").write_text(text, encoding="utf-8")
        print(f"{pdf.name}: {len(text)} characters")
