"""
DatasheetQA — Flask Application

A RAG-based Q&A system for solar/inverter datasheets.
Upload PDFs, ask questions in plain English, get grounded answers
with page citations.
"""

import os
import logging
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from config import config
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask App ────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)
app.config["MAX_CONTENT_LENGTH"] = config.max_file_size_mb * 1024 * 1024

# ── Ensure directories exist ────────────────────────────────
os.makedirs(config.upload_folder, exist_ok=True)
os.makedirs(config.index_folder, exist_ok=True)

# ── Initialize modules ──────────────────────────────────────
pdf_processor = PDFProcessor()
vector_store = VectorStore()
rag_pipeline = RAGPipeline(vector_store)

# ── Try to load existing index ──────────────────────────────
if vector_store.load(config.index_folder, "datasheet_index"):
    logger.info("Loaded existing vector index on startup.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/")
def index():
    """Serve the main application page."""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """
    Upload and process one or more PDF files.

    Extracts text, chunks it, generates embeddings,
    and adds to the vector index.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "No files selected"}), 400

    results = []
    all_chunks = list(vector_store.chunks)  # preserve existing chunks

    for file in files:
        filename = secure_filename(file.filename)

        # Validate extension
        if not filename.lower().endswith(config.allowed_extensions):
            results.append({
                "filename": filename,
                "status": "error",
                "message": "Only PDF files are allowed.",
            })
            continue

        # Save file
        filepath = os.path.join(config.upload_folder, filename)
        file.save(filepath)

        try:
            # Process PDF
            chunks = pdf_processor.process_pdf(filepath, filename)
            all_chunks.extend(chunks)

            results.append({
                "filename": filename,
                "status": "success",
                "chunks": len(chunks),
                "message": f"Processed successfully: {len(chunks)} chunks created.",
            })
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results.append({
                "filename": filename,
                "status": "error",
                "message": str(e),
            })

    # Rebuild index with all chunks (existing + new)
    if all_chunks:
        try:
            vector_store.build_index(all_chunks)
            vector_store.save(config.index_folder, "datasheet_index")
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return jsonify({
                "error": f"Index building failed: {e}",
                "file_results": results,
            }), 500

    return jsonify({
        "results": results,
        "total_chunks": len(all_chunks),
        "documents": vector_store.get_document_list(),
    })


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    Answer a question using the RAG pipeline.

    Expects JSON: {"question": "..."}
    Returns structured JSON with answer, citations, and metadata.
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    if vector_store.index is None or vector_store.index.ntotal == 0:
        # Try reloading from disk (handles Flask reloader restarts)
        vector_store.load(config.index_folder, "datasheet_index")

    if vector_store.index is None or vector_store.index.ntotal == 0:
        return jsonify({
            "error": "No documents indexed. Please upload PDF datasheets first."
        }), 400

    try:
        result = rag_pipeline.answer_question(question)
        result["timestamp"] = datetime.now().isoformat()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """List all indexed documents."""
    # Auto-reload from disk if index was lost (e.g. Flask reloader restart)
    if not vector_store.chunks:
        vector_store.load(config.index_folder, "datasheet_index")

    documents = vector_store.get_document_list()
    total_chunks = len(vector_store.chunks)
    return jsonify({
        "documents": documents,
        "total_chunks": total_chunks,
    })


@app.route("/api/documents", methods=["DELETE"])
def clear_documents():
    """Clear all indexed documents and reset the vector store."""
    vector_store.clear()

    # Remove saved index files
    for fname in os.listdir(config.index_folder):
        fpath = os.path.join(config.index_folder, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    # Remove uploaded files
    for fname in os.listdir(config.upload_folder):
        fpath = os.path.join(config.upload_folder, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    return jsonify({"message": "All documents and indices cleared."})


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    nvidia_ready = bool(config.nvidia_api_key)
    gemini_ready = bool(config.gemini_api_key)
    return jsonify({
        "status": "healthy",
        "documents_loaded": len(vector_store.get_document_list()),
        "total_chunks": len(vector_store.chunks),
        "primary_model":  config.nvidia.model_name if nvidia_ready else "(disabled)",
        "fallback_model": config.generation.model_name if gemini_ready else "(disabled)",
        "nvidia_ready":   nvidia_ready,
        "gemini_ready":   gemini_ready,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    logger.info("Starting DatasheetQA server...")
    logger.info(f"Primary  LLM : NVIDIA NIM — {config.nvidia.model_name}")
    logger.info(f"Fallback LLM : Gemini     — {config.generation.model_name}")
    logger.info(f"Embedding    : {config.embedding.model_name}")
    logger.info(f"NVIDIA key   : {'✓ set' if config.nvidia_api_key else '✗ MISSING'}")
    logger.info(f"Gemini key   : {'✓ set' if config.gemini_api_key else '✗ MISSING'}")
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
    )
