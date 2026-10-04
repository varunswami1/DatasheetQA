# DatasheetQA — AI-Powered Datasheet Intelligence

A **Retrieval-Augmented Generation (RAG)** web application that lets you upload solar panel, inverter, or battery datasheets (PDF) and ask questions in plain English. Get precise, cited answers grounded in your documents.

## Architecture

```
PDF Upload → Chunking → Embeddings (NVIDIA nemotron-embed) → FAISS Index
                                                                    ↓
User Query → Query Embedding → Vector Search → Top-K Chunks → LLM (NVIDIA NIM)
                                                                    ↓
                                                          Answer + Citations
```

**Primary LLM:** `meta/llama-4-scout-17b-16e-instruct` via NVIDIA NIM  
**Fallback LLM:** Google Gemini  
**Embeddings:** `nvidia/nemotron-3-embed-1b` via NVIDIA NIM  
**Vector Store:** FAISS (local, in-memory)

## Features

- 📄 Upload multiple PDF datasheets
- 🔍 Semantic search over document content
- 🤖 LLM-powered answers with page-level citations
- ⚡ Fast inference via NVIDIA NIM hosted endpoints
- 🔄 Automatic Gemini fallback if NVIDIA is unavailable
- 💬 Chat-style interface with conversation history

## Project Structure

```
DatasheetQA/
├── app.py              # Flask server & API routes
├── rag_pipeline.py     # RAG orchestration (retrieval + generation)
├── vector_store.py     # FAISS vector store management
├── pdf_processor.py    # PDF parsing & text chunking
├── prompts.py          # LLM prompt templates
├── config.py           # All tunable configuration parameters
├── evaluate.py         # Evaluation utilities
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── templates/
│   └── index.html      # Frontend UI (Jinja2)
└── static/
    ├── style.css        # UI styles
    └── app.js           # Frontend JavaScript
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/varunswami1/DatasheetQA.git
cd DatasheetQA
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your API keys:
```
NVIDIA_API_KEY=nvapi-...
GEMINI_API_KEY=AIza...
```

Get your keys:
- **NVIDIA NIM:** https://build.nvidia.com → top-right → Get API Key
- **Gemini:** https://aistudio.google.com/app/apikey

### 5. Run the application
```bash
python app.py
```

Open your browser at **http://localhost:5000**

## Configuration

All parameters are centralized in [`config.py`](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `meta/llama-4-scout-17b-16e-instruct` | Primary LLM |
| `temperature` | `0.2` | Generation temperature |
| `max_tokens` | `1024` | Max response tokens |
| `chunk_size` | `800` | Characters per chunk |
| `top_k` | `5` | Retrieved chunks per query |
| `similarity_threshold` | `0.35` | Minimum cosine similarity |

## Technologies Used

- **Backend:** Python, Flask
- **LLM:** NVIDIA NIM (Meta Llama 4 Scout), Google Gemini
- **Embeddings:** NVIDIA nemotron-embed via NIM
- **Vector Search:** FAISS
- **PDF Parsing:** PyMuPDF (fitz)
- **Frontend:** HTML, CSS (Vanilla), JavaScript

## License

MIT
