
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.rag_chat import ask_question
from backend.create_embeddings import create_embeddings


app = FastAPI(
    title="AI Enterprise Knowledge Assistant",
    version="1.0.0"
)


# CORS will be restricted to the local frontend in a later step.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# Project folders
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DOCUMENTS_FOLDER = (PROJECT_FOLDER / "documents").resolve()
DOCUMENTS_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB


def validate_filename(filename: str) -> str:
    """Allow simple .txt filenames only."""

    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._ -]*\.txt",
            filename,
            flags=re.IGNORECASE,
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Use a simple .txt filename."
        )

    return filename


def get_document_path(filename: str) -> Path:
    """Resolve a safe document path inside the documents folder."""

    filename = validate_filename(filename)
    path = (DOCUMENTS_FOLDER / filename).resolve()

    if path.parent != DOCUMENTS_FOLDER:
        raise HTTPException(
            status_code=400,
            detail="Invalid document path."
        )

    return path


# Request models
class ConversationMessage(BaseModel):
    role: str
    content: str


class Question(BaseModel):
    question: str
    conversation: list[ConversationMessage] = Field(
        default_factory=list
    )


# Home
@app.get("/")
def home():
    return {
        "message": "AI Enterprise Knowledge Assistant is running"
    }


# Ask a question
@app.post("/ask")
def ask(question: Question):
    try:
        conversation = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in question.conversation
        ]

        result = ask_question(
            question.question,
            conversation
        )

        return {
            "question": question.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception:
        # Avoid returning internal exception details to the client.
        raise HTTPException(
            status_code=500,
            detail="Unable to process your question. Please try again."
        )


# Upload a text document
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    filename = validate_filename(file.filename)
    file_path = get_document_path(filename)

    try:
        content = await file.read(MAX_UPLOAD_SIZE + 1)

        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 5 MB."
            )

        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        if file_path.exists():
            raise HTTPException(
                status_code=409,
                detail="A document with this filename already exists. Delete it before uploading a replacement."
            )

        file_path.write_bytes(content)

        try:
            create_embeddings()
        except Exception:
            # Remove the newly uploaded file if indexing fails.
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="Document indexing failed. Please try again."
            )

        return {
            "message": "Document uploaded and indexed successfully.",
            "filename": filename
        }

    finally:
        await file.close()


# List documents
@app.get("/documents")
def list_documents():
    documents = [
        {"filename": file.name}
        for file in DOCUMENTS_FOLDER.iterdir()
        if file.is_file() and file.suffix.lower() == ".txt"
    ]

    documents.sort(key=lambda item: item["filename"].lower())

    return {
        "documents": documents,
        "total": len(documents)
    }


# Delete a document
@app.delete("/documents/{filename}")
def delete_document(filename: str):
    file_path = get_document_path(filename)

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    # Keep the document if re-indexing fails.
    backup_content = file_path.read_bytes()
    file_path.unlink()

    try:
        create_embeddings()
    except Exception:
        file_path.write_bytes(backup_content)

        try:
            create_embeddings()
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail="Document deletion could not be completed safely. Please check the server."
        )

    return {
        "message": "Document deleted successfully.",
        "filename": filename
    }
