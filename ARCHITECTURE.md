# 🏗️ AI Enterprise Knowledge Assistant — Architecture

## System Architecture

```mermaid
flowchart TD

    A[👤 Employee / User] --> B[🌐 Frontend<br/>HTML + CSS + JavaScript]

    B --> C[⚡ FastAPI Backend]

    C --> D{Request Type}

    D -->|Ask Question| E[🔎 RAG Pipeline]

    D -->|Upload Document| F[📄 Document Processing]

    D -->|List Documents| G[📚 Document Manager]

    F --> H[✂️ Text Chunking]

    H --> I[🧠 Sentence Transformer<br/>all-MiniLM-L6-v2]

    I --> J[(🗄️ ChromaDB<br/>Vector Database)]

    E --> K[🧠 Query Embedding<br/>all-MiniLM-L6-v2]

    K --> J

    J --> L[📌 Relevant Document Chunks]

    L --> M[📝 Prompt + Retrieved Context]

    M --> N[🤖 Ollama<br/>Llama 3.2 3B]

    N --> O[💬 Generated Answer]

    O --> P[📑 Source Citations]

    P --> B

    F --> I

    G --> Q[(📁 Company Documents)]

    Q --> F