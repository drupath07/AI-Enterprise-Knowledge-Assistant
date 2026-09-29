from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# --------------------------------
# Project folder
# --------------------------------

project_folder = Path(__file__).resolve().parent.parent

documents_folder = project_folder / "documents"

vectorstore_folder = project_folder / "vectorstore"


# --------------------------------
# Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# Create embeddings function
# --------------------------------

def create_embeddings():

    # Connect to ChromaDB

    chroma_client = chromadb.PersistentClient(
        path=str(vectorstore_folder)
    )

    # Get or create collection

    collection = chroma_client.get_or_create_collection(
        name="company_documents"
    )

    # Get all text documents

    documents = list(documents_folder.glob("*.txt"))

    print("Documents found:", len(documents))


    # --------------------------------
    # Clear old embeddings
    # --------------------------------

    existing = collection.get()

    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )


    total_chunks = 0


    # --------------------------------
    # Process documents
    # --------------------------------

    for file_path in documents:

        print(f"Processing: {file_path.name}")


        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()


        # Split document into chunks

        chunks = text.split("\n\n")


        # --------------------------------
        # Create embeddings
        # --------------------------------

        for chunk in chunks:

            chunk = chunk.strip()

            if not chunk:
                continue


            embedding = model.encode(
                chunk
            ).tolist()


            collection.add(

                ids=[
                    f"{file_path.name}_{total_chunks}"
                ],

                documents=[
                    chunk
                ],

                embeddings=[
                    embedding
                ],

                metadatas=[
                    {
                        "source": file_path.name
                    }
                ]

            )


            total_chunks += 1


    print("Embeddings created successfully!")

    print(
        "Total chunks stored:",
        collection.count()
    )


# --------------------------------
# Run directly
# --------------------------------

if __name__ == "__main__":

    create_embeddings()