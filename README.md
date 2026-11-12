# Task 4: Retrieval-Augmented Generation (RAG) Document Chat App

A Streamlit web application that enables users to upload multi-format documents (PDF, TXT, DOCX) and ask questions grounded strictly in their uploaded files using Retrieval-Augmented Generation (RAG) and the Google Gemini API (via the OpenAI-compatible endpoint).

Live app: [to be filled]

---

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that connects Large Language Models (LLMs) to external knowledge sources without retraining or fine-tuning them. Instead of relying solely on parametric knowledge memorized during pretraining, RAG retrieves relevant excerpts from your uploaded documents and provides them directly to the model as grounded context. This approach enables the model to answer domain-specific questions with high factual accuracy, cite exact sources, and eliminate hallucination.

---

## RAG Pipeline Architecture

The application strictly implements the following end-to-end RAG workflow:

```text
+-------------------+
|  1. Document      |  Upload PDF, TXT, or DOCX files and extract text.
+---------+---------+
          |
          v
+-------------------+
|  2. Chunking      |  Split text into overlapping chunks at paragraph/sentence boundaries.
+---------+---------+
          |
          v
+-------------------+
|  3. Embeddings    |  Convert text chunks into dense vectors using gemini-embedding-001.
+---------+---------+
          |
          v
+-------------------+
|  4. Vector DB     |  Store embeddings, chunk text, and source metadata in ChromaDB.
+---------+---------+
          |
          v
+-------------------+
|  5. Retrieval     |  Embed user question and retrieve top-K chunks using cosine distance.
+---------+---------+
          |
          v
+-------------------+
|  6. LLM Context   |  Build grounded prompt with context chunks and anti-hallucination rules.
+---------+---------+
          |
          v
+-------------------+
|  7. Answer        |  Generate grounded answer with interactive source chunk citations.
+-------------------+
```

---

## Features

1. **Multi-Format Document Parsing**:
   - Accepts `.pdf` (via `pypdf`), `.docx` (via `python-docx`), and `.txt` files.
   - Supports multiple files uploaded simultaneously.
   - Shows user-friendly warnings for unreadable or empty files (such as scanned PDFs without OCR text).

2. **Custom Boundary-Aware Chunking**:
   - User-configurable chunk size (`200` to `2000` characters, default `800`).
   - User-configurable overlap (`0` to `400` characters, default `100`).
   - Splits text while prioritizing paragraph (`\n\n`), newline (`\n`), sentence (`. `, `! `, `? `), and word boundaries.
   - Displays real-time chunk counts for each uploaded file.

3. **Batched Embeddings with Backoff**:
   - Embeds text chunks in batches of 20 with `gemini-embedding-001`.
   - Includes automatic exponential backoff retries for rate limits, network timeouts, and server errors.
   - Pauses briefly between batches to respect Google Gemini free-tier rate limits.
   - Displays an interactive Streamlit progress bar during index building.

4. **In-Memory Vector Search via ChromaDB**:
   - Uses `chromadb.EphemeralClient` with cosine similarity (`metadata={"hnsw:space": "cosine"}`).
   - Stores chunk text, precomputed embeddings, source filenames, and chunk indices.
   - "Build / rebuild index" button builds the collection and stores it in `st.session_state` to survive reruns.

5. **Semantic Top-K Retrieval**:
   - Configurable `top_k` slider (`1` to `10`, default `4`).
   - Converts vector distance to intuitive similarity percentage (`similarity = 1 - distance`).

6. **Grounded Generation & Source Citations**:
   - Enforces a strict system prompt: *"Answer ONLY using the context below. If the answer is not in the context, say: I could not find that in the uploaded documents. Do not use outside knowledge."*
   - Low temperature (`0.2`) for deterministic, factual outputs.
   - Expandable source citation card under every answer displaying source file, chunk index, similarity score, and excerpt.
   - Sidebar toggle to show or hide retrieved chunk details.

7. **Safety & Session Controls**:
   - Blocks questions with an informative guide until an index is built.
   - Session guard (`MAX_QUESTIONS_PER_SESSION = 30`) to protect API quotas.
   - "Clear chat" button to reset conversational history.

---

## How to Run Locally

### 1. Prerequisites
- Python 3.9 or higher.
- A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 2. Navigate to Task Folder
```bash
cd Codomax_Generative_AI_Foundations/Task_04_RAG_Chat_With_Documents
```

### 3. Create & Activate a Virtual Environment
```bash
# Windows (PowerShell or Command Prompt)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up Your API Key
Create a `.env` file based on `.env.example`:
```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```
Edit `.env` and add your key:
```env
GEMINI_API_KEY=AIzaSyYourActualKeyHere
```

### 6. Run the Application
```bash
python -m streamlit run app.py
```

---

## Limitations

- **Scanned PDFs Unsupported**: PDFs without an embedded digital text layer (e.g., scanned images without OCR) yield empty text and cannot be indexed.
- **Free-Tier Rate Limits**: Google AI Studio free tier limits requests-per-minute (RPM) and requests-per-day (RPD). Processing very large document sets may experience throttling delays.
- **Dependency on Chunk Size & Top-K**: Answer quality depends directly on chunk size and `top_k`. Chunks that are too small split essential context; chunks that are too large dilute semantic relevance.
- **Index Lost on App Restart**: The ChromaDB vector database runs in memory (`EphemeralClient`). When the Streamlit application restarts or the browser tab refreshes, the index must be rebuilt.

---

## Evaluation Notes

[to be filled after testing]
