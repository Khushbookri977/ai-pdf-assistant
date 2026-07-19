# AI PDF Assistant

Chat with any PDF using a fully local LLM — no API keys, no internet required, no costs.
Upload a document and ask questions in natural language.

---

## Features

- 100% local — no OpenAI, no Gemini, no API costs
- Supports multiple local models (Llama 3.1, Mistral, Qwen2.5, Code Llama)
- RAG pipeline — answers come only from your document, not general knowledge
- Shows which chunks of the document were used to generate each answer
- Adjustable chunk size, overlap, and retrieval count via sidebar
- Chat history within session
- Works offline after first setup

---

## Stack

- Ollama — Local LLM runtime (runs Llama 3.1, Mistral, and others on your machine)
- LangChain — Orchestration and RAG pipeline
- HuggingFace Sentence Transformers — Free local embeddings (all-MiniLM-L6-v2)
- FAISS — Vector store for semantic search
- Streamlit — Web UI
- pypdf — PDF text extraction

---

## Architecture

PDF Upload
  to pypdf which extracts text
  to LangChain which splits into chunks (1000 chars, 200 overlap)
  to HuggingFace all-MiniLM-L6-v2 which converts chunks to vectors (local, free, unlimited)
  to FAISS which stores vectors in memory
  to User asks a question
  to Question embedded and matched against FAISS
  to Top K matching chunks sent to local LLM via Ollama
  to LLM returns answer based only on those chunks
  to Answer displayed in Streamlit UI

---

## Setup

Step 1 — Install Ollama

    Download from https://ollama.com and install it.

Step 2 — Pull a local model

    ollama pull llama3.1:8b

Step 3 — Start the Ollama server

    ollama serve

Step 4 — Clone the repo

    git clone https://github.com/Khushbookri977/ai-pdf-assistant.git
    cd ai-pdf-assistant

Step 5 — Create virtual environment and install dependencies

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Step 6 — Run the app (in a new terminal with venv active)

    streamlit run app.py

Step 7 — Open your browser at http://localhost:8501

---

## Available Models

You can switch models from the sidebar dropdown. Pull any of these with Ollama:

    ollama pull llama3.1:8b     # Best for general Q&A (recommended)
    ollama pull mistral:7b      # Fast, good for document Q&A
    ollama pull qwen2.5:7b      # Strong reasoning
    ollama pull codellama:7b    # Best for code-related PDFs

All models require approximately 8 GB RAM and 5 GB disk space.

---

## Usage

1. Make sure Ollama is running (ollama serve)
2. Open the app at http://localhost:8501
3. Select your preferred model from the sidebar
4. Upload a PDF using the file uploader
5. Wait for text extraction and embedding (takes a few seconds)
6. Type your question in the chat box
7. The LLM answers based only on the content of your PDF
8. Expand Source chunks to see which parts of the document were used

---

## Project Structure

    ai-pdf-assistant/
    ├── app.py                  # Main Streamlit application
    ├── requirements.txt        # Python dependencies
    ├── .gitignore
    ├── LICENSE
    └── README.md

---

## License

Distributed under the GNU General Public License v3.0.
See LICENSE for more information.
