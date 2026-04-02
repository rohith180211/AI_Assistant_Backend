import os
import uuid
import chromadb
import ollama
from fastapi import FastAPI, UploadFile, File
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from fastapi import BackgroundTasks
from pydantic import BaseModel
from database import engine, Base, SessionLocal
from models import Document

Base.metadata.create_all(bind=engine)


model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="documents")

app = FastAPI()

UPLOAD_DIR = "uploads"


@app.get("/")
def read_root():
    return {"message": "AI Knowledge System is running"}


def process_document(doc_id, file_path):
    db = SessionLocal()

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            ids=[f"{doc_id}_{i}"],
            metadatas=[{"doc_id": doc_id}]
        )

    doc = db.query(Document).filter(Document.id == doc_id).first()
    doc.status = "completed"

    db.commit()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    db = SessionLocal()
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    doc = Document(
        id=doc_id,
        filename=file.filename,
        path=file_path,
        status="processing"
    )

    db.add(doc)
    db.commit()

    # Run in background
    background_tasks.add_task(process_document, doc_id, file_path)

    return {
        "document_id": doc_id,
        "status": "processing"
    }


@app.get("/documents")
def list_documents():
    db = SessionLocal()
    try:
        docs = db.query(Document).all()

        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "path": doc.path,
                "status": doc.status
            }
            for doc in docs
        ]
    finally:
        db.close()

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def get_embeddings(chunks):
    return model.encode(chunks)


class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_documents(request: QueryRequest,k: int = 3):
    query_embedding = model.encode([request.question])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    chunks = results["documents"][0]
    clean_sources = list(dict.fromkeys(chunks))
    answer = generate_answer(request.question, clean_sources)
    return {
        "question": request.question,
        "answer": answer,
        "sources": clean_sources
    }

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are an AI assistant. Answer the question ONLY using the provided context.

    If the answer is not in the context, say "I don't know."

    Provide a clear, structured answer.

    Context:
    {context}

    Question:
    {question}
    """

    response = ollama.chat(
        model='llama3',
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']