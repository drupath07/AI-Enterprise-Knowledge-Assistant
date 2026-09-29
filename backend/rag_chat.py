from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ============================================================
# PROJECT PATH
# ============================================================

project_folder = Path(__file__).resolve().parent.parent


# ============================================================
# EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(project_folder / "vectorstore")
)

collection = chroma_client.get_collection(
    name="company_documents"
)


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(query, conversation=None):

    query = query.strip()

    if conversation is None:
        conversation = []


    # ========================================================
    # EMPTY QUESTION
    # ========================================================

    if not query:

        return {
            "answer": (
                "I don't have enough information in the company "
                "documents to answer that."
            ),
            "sources": []
        }


    query_lower = query.lower()


    # ========================================================
    # DETERMINE DOCUMENT SCOPE
    # ========================================================

    filter_condition = None


    # ========================================================
    # COMPANY LEAVE POLICY
    # ========================================================

    leave_keywords = [

        "leave",

        "annual leave",

        "work from home",

        "wfh",

        "emergency leave",

        "manager approval",

        "manager",

        "approval",

        "days off",

        "leave request"

    ]


    # ========================================================
    # BENEFITS POLICY
    # ========================================================

    benefits_keywords = [

        "benefit",

        "benefits",

        "insurance",

        "health insurance",

        "performance bonus",

        "bonus",

        "learning",

        "development program"

    ]


    # ========================================================
    # HR POLICY
    # ========================================================

    hr_keywords = [

        "hr",

        "human resources",

        "performance review",

        "performance reviews",

        "review frequency",

        "twice a year"

    ]


    # ========================================================
    # IT SECURITY POLICY
    # ========================================================

    security_keywords = [

        "security",

        "cybersecurity",

        "security policy",

        "password",

        "multi-factor",

        "mfa",

        "incident",

        "security incident",

        "unauthorized device"

    ]


    # ========================================================
    # SELECT DOCUMENT
    # ========================================================

    if any(
        keyword in query_lower
        for keyword in leave_keywords
    ):

        filter_condition = {
            "source": "company_policy.txt"
        }

    elif any(
        keyword in query_lower
        for keyword in benefits_keywords
    ):

        filter_condition = {
            "source": "benefits_policy.txt"
        }

    elif any(
        keyword in query_lower
        for keyword in hr_keywords
    ):

        filter_condition = {
            "source": "hr_policy.txt"
        }

    elif any(
        keyword in query_lower
        for keyword in security_keywords
    ):

        filter_condition = {
            "source": "it_security_policy.txt"
        }


    # ========================================================
    # CREATE QUERY EMBEDDING
    # ========================================================

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()


    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    if filter_condition:

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=8,
            where=filter_condition,
            include=[
                "documents",
                "metadatas"
            ]
        )

    else:

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=8,
            include=[
                "documents",
                "metadatas"
            ]
        )


    # ========================================================
    # GET RESULTS
    # ========================================================

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadata_list = results.get(
        "metadatas",
        [[]]
    )[0]


    # ========================================================
    # CHECK INFORMATION
    # ========================================================

    if not documents:

        return {
            "answer": (
                "I don't have enough information in the company "
                "documents to answer that."
            ),
            "sources": []
        }


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_documents = []

    for document in documents:

        if (
            document
            and document not in unique_documents
        ):

            unique_documents.append(
                document
            )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = "\n\n".join(
        unique_documents
    )


    # ========================================================
    # GET SOURCES
    # ========================================================

    sources = []

    for metadata in metadata_list:

        if metadata and "source" in metadata:

            source = metadata["source"]

            if source not in sources:

                sources.append(
                    source
                )


    # ========================================================
    # BUILD CONVERSATION HISTORY
    # ========================================================

    conversation_text = ""

    for message in conversation[-6:]:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        if role == "user":

            conversation_text += (
                f"User: {content}\n"
            )

        elif role == "assistant":

            conversation_text += (
                f"Assistant: {content}\n"
            )


    # ========================================================
    # LLM PROMPT
    # ========================================================

    prompt = f"""
You are an AI Enterprise Knowledge Assistant.

Answer the CURRENT USER QUESTION using ONLY the
COMPANY DOCUMENT INFORMATION.

Use the conversation history only to understand
follow-up questions.

IMPORTANT RULES:

1. Company documents are the source of truth.
2. Use only the provided company document information.
3. Use conversation history to understand follow-up questions.
4. Do not treat previous assistant answers as evidence.
5. Do not use outside knowledge.
6. Do not invent information.
7. Do not guess.
8. Answer the current question directly.
9. Do not include unrelated information.
10. If the requested information is not available, respond exactly:

I don't have enough information in the company documents to answer that.

11. Keep the answer concise and professional.

============================================================
RECENT CONVERSATION
============================================================

{conversation_text}

============================================================
COMPANY DOCUMENT INFORMATION
============================================================

{context}

============================================================
CURRENT USER QUESTION
============================================================

{query}

============================================================
ANSWER
============================================================
"""

    # ========================================================
    # OLLAMA
    # ========================================================

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # ========================================================
    # GET ANSWER
    # ========================================================

    answer = response[
        "message"
    ][
        "content"
    ].strip()


    # ========================================================
    # EMPTY ANSWER
    # ========================================================

    if not answer:

        answer = (
            "I don't have enough information in the company "
            "documents to answer that."
        )


    # ========================================================
    # IMPORTANT:
    # REMOVE SOURCES WHEN INFORMATION IS NOT AVAILABLE
    # ========================================================

    insufficient_answer = (
        "I don't have enough information in the company "
        "documents to answer that."
    )

    if answer == insufficient_answer:

        sources = []


    # ========================================================
    # RETURN
    # ========================================================

    return {
        "answer": answer,
        "sources": sources
    }