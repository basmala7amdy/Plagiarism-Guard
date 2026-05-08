import io
from fastapi import HTTPException

import pdfplumber
from docx import Document


# -----------------------------------------------------
# PDF
# -----------------------------------------------------
def extract_pdf(data: bytes) -> str:
    try:
        text = []

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

        result = "\n".join(text).strip()

        if not result:
            return "Unable to extract text (scanned PDF?)"

        return result

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF error: {str(e)}")


# -----------------------------------------------------
# DOCX
# -----------------------------------------------------
def extract_docx(data: bytes) -> str:
    try:
        doc = Document(io.BytesIO(data))

        text = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        return "\n".join(text).strip()

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"DOCX error: {str(e)}")


# -----------------------------------------------------
# TXT
# -----------------------------------------------------
def extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").strip()


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
async def extract_text(file) -> str:
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ("pdf", "docx", "txt"):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    data = await file.read()

    if not data:
        raise HTTPException(status_code=422, detail="Empty file")

    extractors = {
        "pdf": extract_pdf,
        "docx": extract_docx,
        "txt": extract_txt,
    }

    text = extractors[ext](data)

    if not text:
        raise HTTPException(status_code=422, detail="No text extracted")

    return text
    