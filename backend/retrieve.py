from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# Project folder
project_folder = Path(__file__).resolve().parent.parent

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(
    path=str(project_folder / "vectorstore")
)

# Get collection
collection = chroma_client.get_collection(
    name="company_documents"
)

# Ask question
query = input("Ask a question: ")

# Convert question into embedding
query_embedding = model.encode(query).tolist()

# Search the knowledge base
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\n--- RELEVANT INFORMATION ---")

for i, document in enumerate(results["documents"][0]):
    source = results["metadatas"][0][i]["source"]

    print(f"\nSource: {source}")
    print(document)