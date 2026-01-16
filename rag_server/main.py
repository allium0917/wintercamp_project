from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests
import json
import re

app = FastAPI()

class UploadRequest(BaseModel):
    full_text: str
    chunk_size: int

# 전역 변수
embedding_model = None
chroma_client = None

@app.on_event("startup")
async def startup_event():
    global embedding_model, chroma_client
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    print("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient()
    print("Startup complete!")


# txt 정제
def clean_text(text: str) -> str:
    """
    RAG 품질에 직접 영향
    - 불필요한 줄바꿈 / 공백 제거
    """
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# 의미 단위 chunking + overlap
# 변경: 문단 기준 + 토큰 제한 + overlap
def semantic_chunk(text: str, max_tokens=800, overlap=100):
    encoder = tiktoken.encoding_for_model("gpt-5")

    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for p in paragraphs:
        p_tokens = len(encoder.encode(p))

        if current_tokens + p_tokens > max_tokens:
            chunks.append(current_chunk)

            # overlap 적용
            overlap_tokens = encoder.encode(current_chunk)[-overlap:]
            current_chunk = encoder.decode(overlap_tokens) + "\n" + p
            current_tokens = len(encoder.encode(current_chunk))
        else:
            current_chunk += "\n" + p
            current_tokens += p_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

@app.post("/upload")
def upload(request: UploadRequest):
    global embedding_model, chroma_client

    # 텍스트 정제
    cleaned_text = clean_text(request.full_text)

    # 의미 단위 chunking 사용
    chunk_list = semantic_chunk(
        cleaned_text,
        max_tokens=request.chunk_size,
        overlap=100
    )

    class MyEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return embedding_model.encode(input).tolist()

    collection_name = 'samsung_collection6'

    try:
        samsung_collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=MyEmbeddingFunction()
        )
    except:
        samsung_collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=MyEmbeddingFunction()
        )

    # id + metadata
    id_list = []
    metadatas = []

    for index, chunk in enumerate(chunk_list):
        id_list.append(str(index))
        metadatas.append({
            "chunk_index": index,
            "source": "uploaded_text"
        })

    samsung_collection.add(
        documents=chunk_list,
        ids=id_list,
        metadatas=metadatas
    )

    return {
        "ok": True,
        "chunks": len(chunk_list)
    }


class QueryRequest(BaseModel):
    query: str


@app.post("/answer")
def llm_response(request: QueryRequest):
    global embedding_model, chroma_client

    collection_name = 'samsung_collection6'

    class MyEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return embedding_model.encode(input).tolist()

    samsung_collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=MyEmbeddingFunction()
    )

    retrieved_doc = samsung_collection.query(
        query_texts=[request.query],
        n_results=3
    )

    # top-k 문서
    refer = "\n".join(retrieved_doc['documents'][0])

    url = "http://ollama:11434/api/generate"

    payload = {
        "model": "gemma3:1b",
        "prompt": f"""
You are a business analysis expert in Korea.
Please find answers to users' questions in our *Context*.
If not found, tell the user you cannot find the information.

*Context*
{refer}

*Question*
{request.query}

Answer in Korean:
""",
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    return {
        "response": response.json()["response"]
    }
