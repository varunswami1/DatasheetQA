"""
Vector store module for DatasheetQA.

Manages document embeddings using Google Gemini's embedding API
and FAISS for efficient similarity search.
"""

import os
import json
import pickle
import logging
from typing import Optional

import numpy as np
import faiss
from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-backed vector store with Gemini embeddings.

    Stores document chunks as dense vectors and supports
    similarity search with metadata retrieval.
    """

    def __init__(self):
        self.embedding_cfg = config.embedding
        self.retrieval_cfg = config.retrieval
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list = []
        self._client = self._create_client()

    def _create_client(self):
        """Create the NVIDIA API client for embeddings."""
        api_key = config.nvidia_api_key
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY not set. "
                "Export it as an environment variable before running."
            )
        return OpenAI(api_key=api_key, base_url=config.nvidia.base_url)

    def embed_texts(self, texts: list, task_type: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts using Gemini.

        Args:
            texts: List of text strings to embed.
            task_type: Embedding task type (RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY).

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if task_type is None:
            task_type = self.embedding_cfg.task_type

        # Gemini embedding API supports batching
        batch_size = 50
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            result = self._client.embeddings.create(
                input=batch,
                model=self.embedding_cfg.model_name,
                extra_body={"input_type": task_type}
            )
            
            for item in result.data:
                all_embeddings.append(item.embedding)

        embeddings = np.array(all_embeddings, dtype=np.float32)

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        return embeddings

    def build_index(self, chunks: list):
        """
        Build a FAISS index from document chunks.

        Args:
            chunks: List of chunk dicts with at least a "text" key.
        """
        if not chunks:
            raise ValueError("No chunks provided to build index.")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embed_texts(texts, task_type=self.embedding_cfg.task_type)

        # Build FAISS inner-product index (cosine sim on normalized vectors)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        logger.info(f"FAISS index built with {self.index.ntotal} vectors (dim={dim})")

    def search(self, query: str, top_k: Optional[int] = None) -> list:
        """
        Search the index for chunks most similar to the query.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of chunk dicts with an added "score" key, sorted by relevance.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on empty index.")
            return []

        if top_k is None:
            top_k = self.retrieval_cfg.top_k

        # Embed the query
        query_embedding = self.embed_texts(
            [query],
            task_type=self.embedding_cfg.query_task_type
        )

        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < self.retrieval_cfg.similarity_threshold:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        logger.info(
            f"Query returned {len(results)} results "
            f"(threshold={self.retrieval_cfg.similarity_threshold})"
        )
        return results

    def save(self, directory: str, name: str):
        """Save the FAISS index and chunk metadata to disk."""
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, f"{name}.faiss")
        meta_path = os.path.join(directory, f"{name}_meta.pkl")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Index saved to {index_path}")

    def load(self, directory: str, name: str) -> bool:
        """Load a FAISS index and chunk metadata from disk."""
        index_path = os.path.join(directory, f"{name}.faiss")
        meta_path = os.path.join(directory, f"{name}_meta.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(f"Index loaded: {self.index.ntotal} vectors")
        return True

    def get_document_list(self) -> list:
        """Get the list of unique document names in the index."""
        return list(set(c["document"] for c in self.chunks))

    def clear(self):
        """Clear the index and chunks."""
        self.index = None
        self.chunks = []
