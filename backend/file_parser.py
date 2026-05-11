import io
from fastapi import HTTPException
import pdfplumber
from docx import Document


def extract_pdf(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)  # fallback "" for blank pages


def extract_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


async def extract_text(file) -> str:
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if ext not in ("pdf", "docx", "txt"):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: .{ext}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    extractors = {"pdf": extract_pdf, "docx": extract_docx, "txt": extract_txt}
    text = extractors[ext](data).strip()

    if not text:
        raise HTTPException(status_code=422, detail="No text found in file.")

    return text
