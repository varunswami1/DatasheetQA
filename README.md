# MUJ-DS-23FE10CDS00449

| Field | Details |
|---|---|
| Name | Varun Swami |
| Registration Number | 23FE10CDS00449 |
| Branch | Data Science |
| Batch | F |
| Project Title | DatasheetQA – AI-Powered Datasheet Intelligence |
| GitHub Username | varunswami1 |
| Training Program | NLP Project |

---

# DatasheetQA — AI-Powered Datasheet Intelligence

A **Retrieval-Augmented Generation (RAG)** web application that lets you upload solar panel, inverter, or battery datasheets (PDF) and ask questions in plain English. Get precise, cited answers grounded in your documents.

## Repository Structure

```
MUJ-DS-23FE10CDS00449/
├── README.md
├── assignments/        # Training assignments
├── notebooks/          # Jupyter notebooks
├── code/               # Practice code
├── resources/          # Screenshots, results, references
├── presentations/      # Project presentation
└── capstone/           # DatasheetQA project (individual capstone)
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
        ├── style.css       # UI styles
        └── app.js          # Frontend JavaScript
```

## Architecture

```
PDF Upload → Chunking → Embeddings (NVIDIA nemotron-embed) → FAISS Index
                                                                    ↓
User Query → Query Embedding → Vector Search → Top-K Chunks → LLM (Gemini / NVIDIA)
                                                                    ↓
                                                          Answer + Citations
```

**Primary LLM:** `gemini-3.8-flash` via Google Gemini API  
**Fallback LLM:** `meta/llama-3.2-11b-vision-instruct` via NVIDIA NIM  
**Embeddings:** `nvidia/nemotron-3-embed-1b` via NVIDIA NIM  
**Vector Store:** FAISS (local, in-memory)

## Features

- 📄 Upload multiple PDF datasheets
- 🔍 Semantic search over document content (FAISS)
- 🤖 LLM-powered answers with page-level citations
- 🔄 Automatic NVIDIA NIM fallback if Gemini is unavailable
- 💬 Chat-style interface with conversation history
- 🌐 Deployed live on Render

## Project Structure

```
DatasheetQA/
├── app.py              # Flask server & API routes
├── rag_pipeline.py     # RAG orchestration (retrieval + generation)
├── vector_store.py     # FAISS vector store with NVIDIA NIM embeddings
├── pdf_processor.py    # PDF parsing & text chunking
├── prompts.py          # LLM prompt templates
├── config.py           # All tunable configuration parameters
├── evaluate.py         # Evaluation module (keyword hit-rate metrics)
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── .env.example        # Environment variable template
├── templates/
│   └── index.html      # Frontend UI (Jinja2)
└── static/
    ├── style.css        # UI styles (WallWidgy-inspired dark theme)
    └── app.js           # Frontend JavaScript
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/varunswami1/MUJ-DS-23FE10CDS00449.git
cd MUJ-DS-23FE10CDS00449/capstone
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
- **Gemini:** https://aistudio.google.com/app/apikey
- **NVIDIA NIM:** https://build.nvidia.com → top-right → Get API Key

### 5. Run the application
```bash
python app.py
```

Open your browser at **http://localhost:5000**

## Configuration

All parameters are centralized in [`capstone/config.py`](capstone/config.py):

| Parameter | Actual Value | Description |
|-----------|-------------|-------------|
| Primary LLM | `gemini-3.8-flash` | Main answer generation model |
| Fallback LLM | `meta/llama-3.2-11b-vision-instruct` | NVIDIA NIM fallback |
| `temperature` | `0.2` | Lower = more precise/factual answers |
| `max_tokens` | `1024` | Max response length |
| `chunk_size` | `800` | Characters per document chunk |
| `chunk_overlap` | `150` | Overlap between adjacent chunks |
| `top_k` | `7` | Chunks retrieved per query |
| `similarity_threshold` | `0.25` | Minimum cosine similarity for retrieval |

## Evaluation

Run the built-in evaluation suite against a ground-truth question set:

```bash
python evaluate.py
```

Outputs per-question keyword hit-rate, latency, confidence, and provider used. Report saved to `eval_report.json`.

## Live Demo

🌐 **[https://datasheetqa.onrender.com](https://datasheetqa.onrender.com)**

> Note: Free-tier Render instances spin down after inactivity — first request may take ~30s to wake up.

## Technologies Used

- **Backend:** Python 3, Flask, Gunicorn
- **Primary LLM:** Google Gemini 3.8 flash
- **Fallback LLM:** NVIDIA NIM (Meta Llama 3.2 11B Vision)
- **Embeddings:** NVIDIA NIM nemotron-3-embed-1b
- **Vector Search:** FAISS
- **PDF Parsing:** PyMuPDF (fitz), pdfplumber
- **Frontend:** HTML5, Vanilla CSS, JavaScript
- **Deployment:** Render (free tier, persistent disk)

## License

MIT
