"""
Configuration file for DatasheetQA application.

All tunable parameters are centralized here for easy modification.
Primary LLM: NVIDIA NIM (build.nvidia.com) — falls back to Gemini if unavailable.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from the project root
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)

print(f"[CONFIG] .env loaded: {ENV_FILE.exists()}")
print(f"[CONFIG] NVIDIA_API_KEY: {'SET' if os.getenv('NVIDIA_API_KEY') else 'NOT SET'}")
print(f"[CONFIG] GEMINI_API_KEY: {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""
    model_name: str = "nvidia/nemotron-3-embed-1b"
    dimension: int = 2048
    task_type: str = "passage"
    query_task_type: str = "query"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 150       # overlap between consecutive chunks
    min_chunk_size: int = 100      # discard chunks smaller than this
    table_chunk_size: int = 1200   # larger chunks for table-heavy content
    separator: str = "\n"


@dataclass
class RetrievalConfig:
    """Configuration for vector search and retrieval."""
    top_k: int = 7                     # number of chunks to retrieve (more = better recall)
    similarity_threshold: float = 0.25  # lowered for denser technical docs
    rerank: bool = True                 # whether to use LLM-based reranking


@dataclass
class NvidiaConfig:
    """Configuration for the NVIDIA NIM primary LLM."""
    base_url: str = "https://integrate.api.nvidia.com/v1"
    # Available models on NVIDIA NIM (build.nvidia.com):
    #   meta/llama-4-scout-17b-16e-instruct    ← PRIMARY: very fast, great for RAG Q&A
    #   meta/llama-3.3-70b-instruct            ← higher quality, still fast on NIM
    #   meta/llama-3.1-8b-instruct             ← fastest, lower quality
    #   nvidia/nemotron-3-super-120b-a12b      ← high quality but very slow (~2 min)
    #   deepseek-ai/deepseek-v4.1-flash        ← slow cold-start, skip
    model_name: str = "meta/llama-4-scout-17b-16e-instruct"
    temperature: float = 0.2      # lower = more precise factual answers from datasheets
    max_tokens: int = 1024
    top_p: float = 0.95
    max_retries: int = 1
    timeout_seconds: float = 45.0

@dataclass
class GenerationConfig:
    """Configuration for Gemini fallback generation."""
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    top_p: float = 0.95
    max_retries: int = 2


@dataclass
class AppConfig:
    """Main application configuration."""
    # API Keys - loaded from environment variables
    nvidia_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("NVIDIA_API_KEY", "")
    )
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )

    # Sub-configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    nvidia: NvidiaConfig = field(default_factory=NvidiaConfig)          # primary
    generation: GenerationConfig = field(default_factory=GenerationConfig)  # fallback

    # File storage — uses /data (Render persistent disk) in production,
    # falls back to local relative paths for local development
    _data_root: str = field(
        default_factory=lambda: "/data" if Path("/data").exists() else "."
    )
    upload_folder: str = field(
        default_factory=lambda: str(Path("/data/uploads") if Path("/data").exists() else Path("uploads"))
    )
    index_folder: str = field(
        default_factory=lambda: str(Path("/data/indices") if Path("/data").exists() else Path("indices"))
    )
    max_file_size_mb: int = 20
    allowed_extensions: tuple = (".pdf",)

    # Server — debug off in production (gunicorn sets PORT env var on Render)
    host: str = "0.0.0.0"
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", 5000)))
    debug: bool = field(default_factory=lambda: os.environ.get("RENDER") is None)


# Singleton config instance
config = AppConfig()
