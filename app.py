import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocGenius — Local PDF Chat",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocGenius — Chat with your PDF")
st.caption("Powered by LangChain + FAISS + Ollama (Llama 3.1) — 100% local, no API costs")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Choose your local model:",
        ["llama3.1:8b", "mistral:7b", "qwen2.5:7b", "codellama:7b"],
        index=0
    )
    chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
    top_k = st.slider("Number of chunks to retrieve (k)", 1, 10, 4)
    
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Text is split into chunks")
    st.markdown("3. Chunks are embedded with HuggingFace")
    st.markdown("4. Your question finds similar chunks (FAISS)")
    st.markdown("5. Llama 3.1 answers using only those chunks")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="📖 Reading PDF...")
def extract_text_from_pdf(pdf_bytes):
    """Extract all text from uploaded PDF bytes."""
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n[Page {page_num + 1}]\n{page_text}"
    return text


@st.cache_resource(show_spinner="🔪 Splitting text into chunks...")
def split_text(text, _chunk_size, _chunk_overlap):
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


@st.cache_resource(show_spinner="🧠 Creating embeddings (first time is slow)...")
def build_vector_store(chunks):
    """
    Embed all chunks using a free local HuggingFace model.
    all-MiniLM-L6-v2 is small (~90MB), fast, and downloads automatically.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store, embeddings


def build_qa_chain(vector_store, model_name, k):
    """Build a RetrievalQA chain using local Ollama LLM."""
    
    # Custom prompt that tells the model to ONLY use the document
    prompt_template = """You are a helpful assistant that answers questions about a document.
Use ONLY the following context from the document to answer the question.
If the answer is not found in the context, say: "I couldn't find that information in the document."
Do NOT use any prior knowledge outside the provided context.

Context from document:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = ChatOllama(
        model=model_name,
        temperature=0,           # 0 = deterministic, best for Q&A
        num_predict=512,         # max tokens in response
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",      # "stuff" = put all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


# ─────────────────────────────────────────────
# MAIN UI — FILE UPLOAD
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    help="Upload any PDF — resume, textbook, research paper, contract, etc."
)

if uploaded_file is not None:
    # Show file info
    col1, col2, col3 = st.columns(3)
    col1.metric("File name", uploaded_file.name)
    col2.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")
    
    # Read PDF bytes
    pdf_bytes = uploaded_file.read()
    
    # Extract text
    raw_text = extract_text_from_pdf(pdf_bytes)
    
    if not raw_text.strip():
        st.error("❌ Could not extract text from this PDF. It may be a scanned image PDF.")
        st.info("💡 Tip: For scanned PDFs, you'd need OCR (like Tesseract). This app works best with text-based PDFs.")
        st.stop()
    
    col3.metric("Characters extracted", f"{len(raw_text):,}")
    
    with st.expander("👀 Preview extracted text (first 1000 characters)"):
        st.text(raw_text[:1000] + "...")
    
    # Split into chunks
    chunks = split_text(raw_text, chunk_size, chunk_overlap)
    st.success(f"✅ Split into **{len(chunks)} chunks** (size={chunk_size}, overlap={chunk_overlap})")
    
    # Build vector store
    vector_store, embeddings = build_vector_store(tuple(chunks))
    st.success("✅ Embeddings created and stored in FAISS")
    
    # ─────────────────────────────────────────────
    # CHAT INTERFACE
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Ask questions about your PDF")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input box
    if user_question := st.chat_input("Ask anything about the PDF..."):
        
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"🤔 Thinking with {model_choice}..."):
                try:
                    # Build QA chain
                    qa_chain = build_qa_chain(vector_store, model_choice, top_k)
                    
                    # Run the query
                    result = qa_chain.invoke({"query": user_question})
                    
                    answer = result["result"]
                    source_docs = result["source_documents"]
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show source chunks used
                    with st.expander(f"📚 Source chunks used ({len(source_docs)} chunks retrieved)"):
                        for i, doc in enumerate(source_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.markdown("---")
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = str(e)
                    if "connection refused" in error_msg.lower() or "connect" in error_msg.lower():
                        st.error("❌ Cannot connect to Ollama. Make sure Ollama is running!")
                        st.code("ollama serve", language="bash")
                        st.info(f"Then make sure you've pulled the model: `ollama pull {model_choice}`")
                    else:
                        st.error(f"❌ Error: {error_msg}")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.rerun()

else:
    # Landing state — no PDF uploaded yet
    st.info("👆 Upload a PDF above to get started.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📋 Try with your Resume")
        st.markdown("Ask: *What skills does this person have?*")
    with col2:
        st.markdown("#### 📚 Try with a Textbook")
        st.markdown("Ask: *Summarize chapter 3*")
    with col3:
        st.markdown("#### 📑 Try with a Contract")
        st.markdown("Ask: *What are the payment terms?*")