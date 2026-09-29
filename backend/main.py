from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import networkx as nx

from backend.rag_chat import ask_question
from backend.create_embeddings import create_embeddings


app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PROJECT FOLDERS
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

documents_folder = project_folder / "documents"

vectorstore_folder = project_folder / "vectorstore"

documents_folder.mkdir(exist_ok=True)

vectorstore_folder.mkdir(exist_ok=True)


# ============================================================
# REQUEST MODELS
# ============================================================

class ConversationMessage(BaseModel):

    role: str

    content: str


class Question(BaseModel):

    question: str

    conversation: list[ConversationMessage] = []


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Enterprise Knowledge Assistant is running"
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask(question: Question):

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


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    if not file.filename.lower().endswith(".txt"):

        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported"
        )

    file_path = documents_folder / file.filename

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    create_embeddings()

    return {
        "message":
            "Document uploaded and indexed successfully",
        "filename":
            file.filename
    }


# ============================================================
# LIST DOCUMENTS
# ============================================================

@app.get("/documents")
def list_documents():

    documents = []

    for file in documents_folder.iterdir():

        if file.is_file():

            documents.append({
                "filename":
                    file.name
            })

    documents.sort(
        key=lambda item: item["filename"].lower()
    )

    return {
        "documents":
            documents,
        "total":
            len(documents)
    }


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete("/documents/{filename}")
def delete_document(filename: str):

    file_path = documents_folder / filename

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    if not file_path.is_file():

        raise HTTPException(
            status_code=400,
            detail="Invalid document"
        )

    file_path.unlink()

    create_embeddings()

    return {
        "message":
            "Document deleted successfully",
        "filename":
            filename
    }


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

@app.get("/knowledge-graph")
def get_knowledge_graph():

    graph_file = (
        vectorstore_folder
        / "company_knowledge_graph.graphml"
    )

    if not graph_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Knowledge graph not found"
        )

    graph = nx.read_graphml(
        graph_file
    )

    nodes = []

    for node, data in graph.nodes(data=True):

        nodes.append({
            "id": node,
            "type": data.get(
                "type",
                "unknown"
            )
        })

    relationships = []

    for source, target, data in graph.edges(
        data=True
    ):

        relationships.append({
            "source": source,
            "target": target,
            "relation": data.get(
                "relation",
                "related_to"
            )
        })

    return {
        "nodes": nodes,
        "relationships": relationships,
        "total_nodes": len(nodes),
        "total_relationships": len(relationships)
    }