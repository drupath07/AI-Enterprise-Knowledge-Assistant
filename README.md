# 🤖 AI Enterprise Knowledge Assistant

An AI-powered enterprise knowledge assistant that allows employees to ask questions about company policies and documents using Natural Language.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from company documents and generate accurate answers using a locally hosted Large Language Model.

---

## 📌 Project Overview

The AI Enterprise Knowledge Assistant provides a conversational interface for employees to interact with internal company documents.

Instead of manually searching through multiple company policy documents, users can ask questions such as:

- How many days of annual leave do employees get?
- How many days of work from home are allowed?
- What should employees do for emergency leave?
- What employee benefits are available?
- Are employees eligible for a performance bonus?
- How often are performance reviews conducted?
- What should employees do if they suspect a security incident?

The assistant retrieves relevant information from the company's document collection and generates an answer based only on the available documents.

---

# 🎯 Objectives

The main objectives of this project are:

1. Build an AI-powered enterprise knowledge assistant.
2. Allow users to upload company documents.
3. Convert documents into vector embeddings.
4. Store embeddings in a vector database.
5. Retrieve relevant information using semantic search.
6. Generate answers using an LLM.
7. Display the sources used to generate each answer.
8. Prevent the system from inventing information that is not present in the documents.
9. Evaluate the accuracy of the RAG system using predefined test cases.

---

# 🏗️ System Architecture

```text
                  ┌──────────────────────┐
                  │       User           │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Web Frontend       │
                  │     HTML/CSS/JS      │
                  └──────────┬───────────┘
                             │
                             │ HTTP Requests
                             ▼
                  ┌──────────────────────┐
                  │      FastAPI         │
                  │       Backend       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      RAG Pipeline    │
                  │                      │
                  │  Query Processing    │
                  │        ↓             │
                  │  Query Embedding     │
                  │        ↓             │
                  │ Semantic Retrieval   │
                  │        ↓             │
                  │ Context Creation     │
                  │        ↓             │
                  │     LLM Answer       │
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │    ChromaDB     │          │     Ollama      │
     │ Vector Database  │          │  Llama 3.2 3B  │
     └─────────────────┘          └─────────────────┘
              ▲
              │
              │ Embeddings
              │
     ┌─────────────────┐
     │ Company         │
     │ Documents       │
     │ (.txt files)    │
     └─────────────────┘