# AI PDF Assistant

Chat with any PDF using Google Gemini (free tier) — upload a document and ask questions in natural language. Built with Python, LangChain, FAISS, and Streamlit.

---

## Features

- Upload any PDF and ask questions in plain English
- Answers generated using Google Gemini 2.0 Flash (free, no credit card needed)
- RAG pipeline — answers come only from your document, not general knowledge
- Shows which chunks of the document were used to generate each answer
- Adjustable chunk size, overlap, and retrieval count via sidebar
- Chat history within session

---

## Stack

- Google Gemini 2.0 Flash — Free LLM (1,000 requests/day, 15 requests/minute)
- LangChain — Orchestration and RAG pipeline
- FAISS — Vector store for semantic search
- Streamlit — Web UI
- pypdf — PDF text extraction
- python-dotenv — API key management

---

## Architecture

PDF Upload
  → pypdf extracts text
  → LangChain splits into chunks (1000 chars, 200 overlap)
  → Gemini Embeddings converts chunks to vectors
  → FAISS stores vectors locally
  → User asks a question
  → Question embedded and matched against FAISS
  → Top K matching chunks sent to Gemini
  → Gemini returns answer based only on those chunks
  → Answer displayed in Streamlit UI

---

## Setup

Step 1 — Get a free Gemini API key at https://aistudio.google.com

Step 2 — Clone the repo

    git clone https://github.com/Khushbookri977/ai-pdf-assistant.git
    cd ai-pdf-assistant

Step 3 — Create virtual environment and install dependencies

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Step 4 — Add your API key

    Create a file called .env in the project root and add:
    GEMINI_API_KEY=your_key_here

Step 5 — Run the app

    streamlit run app.py

Step 6 — Open your browser at http://localhost:8501

---

## Usage

1. Upload a PDF using the file uploader
2. Wait for text extraction and embedding (takes a few seconds)
3. Type your question in the chat box
4. Gemini answers based only on the content of your PDF
5. Expand "Source chunks" to see which parts of the document were used

---

## Free Tier Limits

- 1,000 requests per day
- 15 requests per minute
- No credit card required
- Data may be used by Google to improve their products on the free tier

---

## Project Structure

    ai-pdf-assistant/
    ├── app.py              # Main Streamlit application
    ├── requirements.txt    # Python dependencies
    ├── .env                # API key (not pushed to GitHub)
    ├── .gitignore
    ├── LICENSE
    └── README.md

---

## License

Distributed under the GNU General Public License v3.0.
See LICENSE for more information.