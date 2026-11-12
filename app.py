"""
Task 4: Retrieval-Augmented Generation (RAG) Document Chat App
==============================================================
A Streamlit web application enabling users to upload documents (PDF, TXT, DOCX)
and ask questions grounded strictly in their uploaded files.

Pipeline:
Document -> Chunking -> Embeddings -> Vector DB -> Retrieval -> LLM -> Answer

Features:
- Google Gemini API via official OpenAI-compatible endpoint.
- Multi-format document text extraction (pypdf, python-docx, utf-8 txt).
- Boundary-aware custom text chunking with configurable size and overlap.
- Batched embedding generation with exponential backoff for free-tier resilience.
- In-memory vector database with ChromaDB and cosine similarity.
- Top-K semantic retrieval with distance-to-similarity conversion.
- Grounded generation with strict anti-hallucination prompt.
- Interactive source citation expanders and session abuse guards.
"""

import os
import io
import time
from typing import Optional, Tuple, Dict, Any, List

import streamlit as st
from dotenv import load_dotenv
import pypdf
import docx
import chromadb
import openai
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    APIError,
    BadRequestError,
    NotFoundError,
)

# ==============================================================================
# PIPELINE CONFIGURATION & CONSTANTS
# ==============================================================================
# Google Gemini API endpoint for OpenAI-compatible client
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Default LLM chat and embedding models
# Verify exact model names in Google AI Studio and edit these if they differ.
CHAT_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL: str = "gemini-embedding-001"

# Maximum questions allowed per session to protect free-tier rate limits and quota
MAX_QUESTIONS_PER_SESSION: int = 30

# Maximum retry attempts for transient API errors
MAX_RETRIES: int = 3

# Request timeout in seconds
REQUEST_TIMEOUT_SECONDS: float = 30.0

# Batch size for embedding chunks
EMBEDDING_BATCH_SIZE: int = 20


# ==============================================================================
# SECTION 1: API KEY RETRIEVAL & CLIENT SETUP
# ==============================================================================
def load_gemini_api_key() -> str:
    """
    Retrieves the Google Gemini API key securely.
    Checks st.secrets first (for Streamlit Community Cloud), then falls back
    to environment variables loaded from .env via python-dotenv.
    Halts execution with a user-friendly error if the key is missing.
    Never prints or hardcodes the key.
    """
    load_dotenv()

    key: Optional[str] = None
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            key = str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    if not key:
        env_val = os.getenv("GEMINI_API_KEY")
        if env_val:
            key = env_val.strip()

    if not key:
        st.error(
            "**Gemini API Key Missing**: Could not find `GEMINI_API_KEY` in Streamlit "
            "secrets or local `.env` file.\n\n"
            "Please create a `.env` file with `GEMINI_API_KEY=your_key_here` or configure "
            "it in `.streamlit/secrets.toml`. You can obtain a free key at https://aistudio.google.com/"
        )
        st.stop()

    return key


# ==============================================================================
# SECTION 2: DOCUMENT PROCESSING (PIPELINE STEP 1)
# ==============================================================================
def extract_text_from_file(uploaded_file) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts text from PDF, DOCX, or TXT file objects.
    Returns (extracted_text, None) on success or (None, error_message) on failure.
    """
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if not file_bytes:
        return None, f"File '{uploaded_file.name}' is empty (0 bytes)."

    try:
        if filename.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text.strip())
            full_text = "\n\n".join(pages_text).strip()
            if not full_text:
                return (
                    None,
                    f"File '{uploaded_file.name}' contains no readable text. "
                    "It may be a scanned image without an OCR layer.",
                )
            return full_text, None

        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs).strip()
            if not full_text:
                return None, f"File '{uploaded_file.name}' contains no readable paragraph text."
            return full_text, None

        elif filename.endswith(".txt"):
            full_text = file_bytes.decode("utf-8", errors="replace").strip()
            if not full_text:
                return None, f"File '{uploaded_file.name}' contains no text."
            return full_text, None

        else:
            return None, f"Unsupported file format for '{uploaded_file.name}'."

    except Exception as exc:
        return None, f"Failed to parse '{uploaded_file.name}': {str(exc)}"


# ==============================================================================
# SECTION 3: BOUNDARY-AWARE CHUNKING (PIPELINE STEP 2)
# ==============================================================================
def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text into chunks by character count with overlap, attempting to break
    cleanly at paragraph (\n\n), newline (\n), sentence (. / ! / ?), or word boundaries.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Look backwards from end for natural break points
            # 1. Paragraph boundary
            p_break = text.rfind("\n\n", start + chunk_overlap, end)
            if p_break != -1:
                end = p_break + 2
            else:
                # 2. Line boundary
                l_break = text.rfind("\n", start + chunk_overlap, end)
                if l_break != -1:
                    end = l_break + 1
                else:
                    # 3. Sentence boundary
                    s_break = -1
                    for punct in [". ", "! ", "? "]:
                        idx = text.rfind(punct, start + chunk_overlap, end)
                        if idx > s_break:
                            s_break = idx
                    if s_break != -1:
                        end = s_break + 2
                    else:
                        # 4. Word boundary
                        w_break = text.rfind(" ", start + chunk_overlap, end)
                        if w_break != -1:
                            end = w_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


# ==============================================================================
# SECTION 4: BATCHED EMBEDDINGS (PIPELINE STEP 3)
# ==============================================================================
def embed_texts_in_batches(
    client: OpenAI,
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    progress_bar=None,
) -> List[List[float]]:
    """
    Embeds a list of texts in batches using client.embeddings.create with retry
    and exponential backoff for transient errors, with a brief sleep between batches.
    """
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, len(texts))
        batch_slice = texts[start_i:end_i]

        embeddings = None
        for attempt in range(MAX_RETRIES):
            try:
                res = client.embeddings.create(
                    model=model,
                    input=batch_slice,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                embeddings = [item.embedding for item in res.data]
                break
            except RateLimitError as exc:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"Gemini API rate limit reached during embedding: {exc}")
            except (APIConnectionError, APITimeoutError) as exc:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"Network error during embedding: {exc}")
            except APIError as exc:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"Gemini API error during embedding: {exc}")
            except Exception as exc:
                raise exc

        if embeddings is None:
            raise Exception("Failed to retrieve embeddings after maximum retry attempts.")

        all_embeddings.extend(embeddings)

        if progress_bar is not None:
            progress_bar.progress((batch_idx + 1) / total_batches)

        # Brief delay between batches to respect free-tier rate limits
        if batch_idx < total_batches - 1:
            time.sleep(0.5)

    return all_embeddings


# ==============================================================================
# SECTION 5: VECTOR DB & RETRIEVAL (PIPELINE STEPS 4 & 5)
# ==============================================================================
def retrieve_relevant_chunks(
    client: OpenAI,
    collection,
    query: str,
    top_k: int = 4,
    embedding_model: str = EMBEDDING_MODEL,
) -> List[Dict[str, Any]]:
    """
    Embeds the user question, queries ChromaDB, and converts cosine distances
    to similarity scores (similarity = 1 - distance).
    """
    # Embed user question
    query_embeddings = embed_texts_in_batches(
        client=client,
        texts=[query],
        model=embedding_model,
        batch_size=1,
    )
    query_vector = query_embeddings[0]

    # Query Chroma collection
    total_docs = collection.count()
    effective_k = min(top_k, total_docs)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: List[Dict[str, Any]] = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
        dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, dists):
            # Cosine distance to similarity: similarity = 1 - distance
            similarity = max(0.0, 1.0 - float(dist))
            retrieved.append({
                "source": meta.get("source", "Unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "similarity": similarity,
                "text": doc_text,
            })

    return retrieved


# ==============================================================================
# SECTION 6: GROUNDED GENERATION (PIPELINE STEPS 6 & 7)
# ==============================================================================
def generate_grounded_answer(
    client: OpenAI,
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_model: str = CHAT_MODEL,
    temperature: float = 0.2,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Constructs a grounded prompt containing retrieved context chunks and calls
    Gemini Chat Completions API with error handling and retries.
    """
    system_prompt = (
        "Answer ONLY using the context below. If the answer is not in the context, "
        "say: I could not find that in the uploaded documents. Do not use outside knowledge."
    )

    # Format numbered context sections
    context_sections = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_sections.append(
            f"[{i}] (Source: {chunk['source']}, Chunk #{chunk['chunk_index']})\n{chunk['text']}"
        )
    formatted_context = "\n\n".join(context_sections)

    user_content = f"Context:\n{formatted_context}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            res = client.chat.completions.create(
                model=chat_model,
                messages=messages,
                temperature=temperature,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            answer = res.choices[0].message.content or ""
            return answer, None

        except AuthenticationError as exc:
            return None, (
                "**Authentication Error**: Invalid Gemini API key. "
                "Please verify your key in Google AI Studio."
            )
        except NotFoundError as exc:
            return None, (
                f"**Model Not Found (404)**: The model `{chat_model}` was not found. "
                "Please verify model identifiers in Google AI Studio."
            )
        except BadRequestError as exc:
            return None, f"**Bad Request (400)**: Request rejected by Gemini API. ({exc})"
        except RateLimitError as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None, (
                "**Rate Limit Error**: Gemini API rate limit reached (free tier). "
                "Please wait a moment before trying again."
            )
        except (APIConnectionError, APITimeoutError) as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None, (
                "**Network Error**: Connection to Google Gemini servers timed out or failed. "
                "Please check your internet connection."
            )
        except APIError as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None, f"**Gemini API Error**: {exc}"
        except Exception as exc:
            return None, f"**Unexpected Error**: {exc}"

    return None, "**Request Failed**: Maximum retry attempts exceeded. Please try again later."


# ==============================================================================
# SECTION 7: STREAMLIT APP UI & WORKSPACE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Chat With Your Documents (RAG)",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chat With Your Documents")
st.caption(
    "Grounded Question Answering powered by Google Gemini and ChromaDB\n"
    "Pipeline: Document -> Chunking -> Embeddings -> Vector DB -> Retrieval -> LLM -> Answer"
)

# Initialize Gemini Client
api_key = load_gemini_api_key()
client = OpenAI(
    api_key=api_key,
    base_url=GEMINI_BASE_URL,
)

# Initialize Session State
if "chroma_collection" not in st.session_state:
    st.session_state.chroma_collection = None

if "index_built" not in st.session_state:
    st.session_state.index_built = False

if "total_chunks_indexed" not in st.session_state:
    st.session_state.total_chunks_indexed = 0

if "files_summary" not in st.session_state:
    st.session_state.files_summary = {}

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==============================================================================
# SECTION 8: SIDEBAR CONTROLS & INDEX BUILDER
# ==============================================================================
st.sidebar.header("Document & Index Settings")

# 1. File Upload
st.sidebar.subheader("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF, TXT, or DOCX files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    help="Upload one or multiple files to build your knowledge base.",
)

# 2. Chunking Settings
st.sidebar.subheader("2. Chunking Settings")
chunk_size = st.sidebar.slider(
    "Chunk Size (characters)",
    min_value=200,
    max_value=2000,
    value=800,
    step=50,
    help="Target size for text segments.",
)
chunk_overlap = st.sidebar.slider(
    "Chunk Overlap (characters)",
    min_value=0,
    max_value=400,
    value=100,
    step=20,
    help="Number of overlapping characters between consecutive chunks.",
)

# 3. Index Building Trigger
st.sidebar.subheader("3. Build Vector Index")
if st.sidebar.button("Build / rebuild index", use_container_width=True, type="primary"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one document first.")
    else:
        extracted_files: Dict[str, str] = {}
        for uploaded_file in uploaded_files:
            text, err = extract_text_from_file(uploaded_file)
            if err:
                st.sidebar.warning(err)
            elif text:
                extracted_files[uploaded_file.name] = text

        if not extracted_files:
            st.sidebar.error("No extractable text found in uploaded files.")
        else:
            all_chunks: List[str] = []
            all_metadatas: List[Dict[str, Any]] = []
            all_ids: List[str] = []
            file_chunk_counts: Dict[str, int] = {}

            # Perform chunking per file
            for fname, ftext in extracted_files.items():
                chunks = chunk_text(ftext, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                file_chunk_counts[fname] = len(chunks)
                for c_idx, c_text in enumerate(chunks):
                    all_chunks.append(c_text)
                    all_metadatas.append({"source": fname, "chunk_index": c_idx})
                    # Unique identifier per chunk
                    all_ids.append(f"{fname}_{c_idx}")

            st.sidebar.markdown("**Chunks Created per File:**")
            for fname, count in file_chunk_counts.items():
                st.sidebar.write(f"- `{fname}`: {count} chunks")

            # Embed chunks with progress bar
            progress_bar = st.sidebar.progress(0.0, text="Generating embeddings...")
            try:
                all_embeddings = embed_texts_in_batches(
                    client=client,
                    texts=all_chunks,
                    model=EMBEDDING_MODEL,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    progress_bar=progress_bar,
                )
                progress_bar.empty()

                # Build Ephemeral ChromaDB Collection
                chroma_client = chromadb.EphemeralClient()
                collection_name = f"rag_collection_{int(time.time())}"
                collection = chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                collection.add(
                    documents=all_chunks,
                    embeddings=all_embeddings,
                    metadatas=all_metadatas,
                    ids=all_ids,
                )

                # Persist in session state
                st.session_state.chroma_collection = collection
                st.session_state.index_built = True
                st.session_state.total_chunks_indexed = len(all_chunks)
                st.session_state.files_summary = file_chunk_counts
                st.sidebar.success(f"Indexed {len(all_chunks)} chunks successfully!")

            except Exception as exc:
                progress_bar.empty()
                st.sidebar.error(f"Indexing failed: {exc}")

# Display current indexing status
if st.session_state.index_built:
    st.sidebar.info(
        f"**Active Index:** {st.session_state.total_chunks_indexed} total chunks indexed."
    )

# 4. Retrieval Settings
st.sidebar.subheader("4. Retrieval Settings")
top_k = st.sidebar.slider(
    "Top K Retrieved Chunks",
    min_value=1,
    max_value=10,
    value=4,
    step=1,
    help="Number of most similar context chunks to supply to the LLM.",
)
show_sources = st.sidebar.toggle(
    "Show retrieved chunks",
    value=True,
    help="Toggles the expandable source citations display under answers.",
)

# 5. Session Actions
st.sidebar.subheader("5. Session Actions")
if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# ==============================================================================
# SECTION 9: CHAT INTERFACE & GENERATION
# ==============================================================================
# Render conversation history
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    sources = msg.get("sources", [])

    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant" and sources and show_sources:
            with st.expander("Sources / retrieved chunks", expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"**Chunk {i}** | *Source:* `{src['source']}` (Chunk #{src['chunk_index']}) | "
                        f"*Similarity:* `{src['similarity']:.2%}`"
                    )
                    st.text(src["text"])

# Session Question Guard
user_questions_count = sum(1 for m in st.session_state.messages if m.get("role") == "user")
is_session_exhausted = user_questions_count >= MAX_QUESTIONS_PER_SESSION

if is_session_exhausted:
    st.warning(
        f"Session limit reached ({MAX_QUESTIONS_PER_SESSION} questions). "
        "Click **Clear chat** in the sidebar to start a new session."
    )

if not st.session_state.index_built:
    st.info("Please upload documents in the sidebar and click **Build / rebuild index** to start chatting.")

chat_disabled = (not st.session_state.index_built) or is_session_exhausted

user_query = st.chat_input(
    "Ask a question about your uploaded documents...",
    disabled=chat_disabled,
)

if user_query and not chat_disabled:
    # 1. Record and render user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Retrieval step
    with st.spinner("Searching documents for relevant context..."):
        try:
            chunks = retrieve_relevant_chunks(
                client=client,
                collection=st.session_state.chroma_collection,
                query=user_query,
                top_k=top_k,
                embedding_model=EMBEDDING_MODEL,
            )
        except Exception as exc:
            st.error(f"Retrieval error: {exc}")
            chunks = []

    # 3. Generation step
    if chunks:
        with st.spinner("Generating grounded answer..."):
            answer, err = generate_grounded_answer(
                client=client,
                query=user_query,
                retrieved_chunks=chunks,
                chat_model=CHAT_MODEL,
                temperature=0.2,
            )

        if err:
            st.error(err)
        elif answer:
            with st.chat_message("assistant"):
                st.markdown(answer)
                if show_sources:
                    with st.expander("Sources / retrieved chunks", expanded=False):
                        for i, src in enumerate(chunks, 1):
                            st.markdown(
                                f"**Chunk {i}** | *Source:* `{src['source']}` (Chunk #{src['chunk_index']}) | "
                                f"*Similarity:* `{src['similarity']:.2%}`"
                            )
                            st.text(src["text"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": chunks,
            })
    else:
        st.warning("No relevant text chunks could be retrieved to answer your question.")
