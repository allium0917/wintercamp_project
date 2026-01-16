import os
import requests

from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
import pymupdf
import pymupdf4llm

app = FastAPI()

# RAG 서버 주소
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://rag_server:8888")

# 로그인용 모델
class LoginUser(BaseModel):
    username: str
    password: str


users = [
    LoginUser(username="park", password="q1w2e3"),
    LoginUser(username="choi", password="q1w2e3"),
]

# 로그인
@app.post("/login")
def login(user: LoginUser = Body()):
    ok = any(u.username == user.username and u.password == user.password for u in users)
    if not ok:
        raise HTTPException(status_code=401, detail="invalid credentials")

    res = JSONResponse({"ok": True})
    res.set_cookie("username", user.username, httponly=True)
    return res


def get_current_user(request: Request) -> str:
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")

    if username not in [u.username for u in users]:
        raise HTTPException(status_code=401, detail="다시 로그인해주세요")

    return username


# PDF 업로드 → RAG 서버 전달
class RagUploadRequest(BaseModel):
    full_text: str
    chunk_size: int = 1000


@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    user = get_current_user(request)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF만 업로드 가능")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="빈 파일")

    # PDF → 텍스트
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        full_text = pymupdf4llm.to_markdown(doc)
    finally:
        doc.close()

    # 👉 RAG 서버로 full_text 전달 (니 코드 그대로 사용)
    payload = {
        "full_text": full_text,
        "chunk_size": 1000
    }

    res = requests.post(f"{RAG_SERVER_URL}/upload", json=payload)
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail="RAG 서버 업로드 실패")

    return {
        "ok": True,
        "user": user,
        "chars": len(full_text),
        "rag": res.json()
    }


# 질문 → RAG 서버로 전달
class QuestionRequest(BaseModel):
    query: str


@app.post("/ask")
def ask_rag(request: Request, body: QuestionRequest):
    user = get_current_user(request)

    res = requests.post(
        f"{RAG_SERVER_URL}/answer",
        json={"query": body.query}
    )

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail="RAG 서버 응답 실패")

    return {
        "ok": True,
        "user": user,
        "answer": res.json()["response"]
    }
