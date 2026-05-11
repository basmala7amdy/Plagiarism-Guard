import html
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from .file_parser import extract_text
from run_detector import predict_plagiarism


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("plagiarism_guard")

app = FastAPI(
    title="Plagiarism Guard API",
    description="AI-powered plagiarism detection service",
    version="1.0.0",
)


class CheckRequest(BaseModel):
    text: str


def build_response(text: str, detection: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = detection.get("results", [])
    sentences = [r.get("sentence", "") for r in results]

    word_count = len(text.split())
    sentence_count = len(sentences)
    unique_phrases = len({s.strip().lower() for s in sentences if s.strip()})
    score = int(round(detection.get("average_score", 0)))

    # Aggregate best score per source document
    sources_by_doc: Dict[str, Dict[str, Any]] = {}
    for sentence in results:
        for detail in sentence.get("details", []):
            doc_id = detail.get("doc_id", "unknown")
            detail_score = float(detail.get("final_score", 0.0))
            current = sources_by_doc.get(doc_id)
            if current is None or detail_score > current["score"]:
                sources_by_doc[doc_id] = {
                    "title": doc_id,
                    "domain": doc_id,
                    "url": detail.get("url", "#") if isinstance(detail.get("url"), str) else "#",
                    "score": detail_score,
                    "excerpt": detail.get("text", "")[:300],
                }

    sources = sorted(sources_by_doc.values(), key=lambda x: x["score"], reverse=True)[:5]  # top 5

    # Build highlighted HTML; escape to prevent XSS
    highlighted_parts = []
    for sentence, result in zip(sentences, results):
        escaped = html.escape(sentence, quote=False)  # prevent XSS in rendered HTML
        if result.get("prediction") == "plagiarism":
            highlighted_parts.append(f'<mark class="plagiarized">{escaped}</mark>')
        else:
            highlighted_parts.append(escaped)
    highlighted_text = "\n\n".join(highlighted_parts)  # double newline for CSS pre-wrap

    matched = float(score)
    original = max(0.0, 100.0 - matched)

    chart_data = {
        "matched": matched,
        "original": original,
        "source_breakdown": [
            {"label": src["title"], "value": round(src["score"] * 100, 2)}
            for src in sources
        ],
        "similarity_timeline": [
            {"segment": idx + 1, "similarity": r.get("sentence_plagiarism_percentage", 0.0)}
            for idx, r in enumerate(results)
        ],
    }

    return {
        "score": score,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_phrases": unique_phrases,
        "sources": sources,
        "highlighted_text": highlighted_text,
        "chart_data": chart_data,
        "results": results,
    }


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning("HTTP error on %s : %s", request.url, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail or "HTTP error"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s : %s", request.url, exc.errors())
    return JSONResponse(status_code=422, content={"error": "Invalid request payload", "details": exc.errors()})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/")
async def root():
    return {"message": "Plagiarism Guard API running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/check")
async def check_text(payload: CheckRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="No text provided.")
    logger.info("Processing text check request")
    result = predict_plagiarism(text)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])
    return build_response(text, result)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("Processing uploaded file: %s", file.filename)
        text = await extract_text(file)
        if not text.strip():
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")
        result = predict_plagiarism(text)
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=422, detail=result["error"])
        return {"filename": file.filename, "result": build_response(text, result)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to process upload request")
        raise HTTPException(status_code=500, detail="Unable to process uploaded file.") from exc