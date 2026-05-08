from fastapi import FastAPI, UploadFile, File
from run_detector import predict_plagiarism
from utils.file_parser import extract_text

app = FastAPI()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    text = await extract_text(file)
    result = predict_plagiarism(text)
    return {"filename": file.filename, "result": result}
