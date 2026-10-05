"""
RAG (Retrieval-Augmented Generation) pipeline for DatasheetQA.

Orchestrates query rewriting, retrieval, context assembly,
LLM generation, and response validation.

LLM Strategy:
  Primary  → Google Gemini 3.8 flash (better synthesis for all question types)
  Fallback → NVIDIA NIM  (meta/llama-3.2-11b-vision-instruct via integrate.api.nvidia.com)
"""

import json
import logging
import re
from typing import Optional

# NVIDIA NIM uses an OpenAI-compatible endpoint
import httpx
# pyrefly: ignore [missing-import]
from openai import OpenAI

from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types

from config import config
from vector_store import VectorStore
from prompts import (
    QUERY_REWRITE_PROMPT,
    QA_PROMPT,
    COMPARISON_PROMPT,
    SYSTEM_MESSAGE,
)

logger = logging.getLogger(__name__)


# ── Provider constants ───────────────────────────────────────────────────────
PROVIDER_NVIDIA = "nvidia"
PROVIDER_GEMINI = "gemini"


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    1. Rewrite query for better retrieval
    2. Retrieve relevant chunks from vector store
    3. Assemble context with metadata
    4. Generate grounded answer — tries NVIDIA first, falls back to Gemini
    5. Validate and parse JSON response
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.nvidia_cfg = config.nvidia
        self.gemini_cfg = config.generation

        # ── NVIDIA NIM client (OpenAI-compatible) ─────────────────────────
        self._nvidia_client: Optional[OpenAI] = None
        if config.nvidia_api_key:
            self._nvidia_client = OpenAI(
                base_url=self.nvidia_cfg.base_url,
                api_key=config.nvidia_api_key,
                # Hard timeout so slow/cold models fail fast → Gemini fallback kicks in
                http_client=httpx.Client(timeout=self.nvidia_cfg.timeout_seconds),
                # Disable internal retries — we handle fallback ourselves
                max_retries=0,
            )
            logger.info(
                f"NVIDIA NIM client initialized — model: {self.nvidia_cfg.model_name} "
                f"(timeout: {self.nvidia_cfg.timeout_seconds}s, no internal retries)"
            )
        else:
            logger.warning(
                "NVIDIA_API_KEY not set — NVIDIA NIM disabled, using Gemini only."
            )

        # ── Gemini client (fallback) ─────────────────────────────────────
        self._gemini_client: Optional[genai.Client] = None
        if config.gemini_api_key:
            self._gemini_client = genai.Client(api_key=config.gemini_api_key)
            logger.info(
                f"Gemini fallback client initialized — model: {self.gemini_cfg.model_name}"
            )
        else:
            logger.warning("GEMINI_API_KEY not set — Gemini fallback disabled.")

    # ── Low-level generation helpers ─────────────────────────────────────────

    def _generate_nvidia(self, prompt: str, force_json: bool = False) -> str:
        """Generate text using the NVIDIA NIM API (OpenAI-compatible)."""
        if not self._nvidia_client:
            raise RuntimeError("NVIDIA NIM client not initialized (missing API key).")

        kwargs = dict(
            model=self.nvidia_cfg.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user",   "content": prompt},
            ],
            temperature=self.nvidia_cfg.temperature,
            max_tokens=self.nvidia_cfg.max_tokens,
            top_p=self.nvidia_cfg.top_p,
        )

        # Force JSON output mode when needed (prevents markdown-wrapped responses)
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._nvidia_client.chat.completions.create(**kwargs)
        except Exception as e:
            # If json_object mode not supported, retry without it
            if force_json and ("response_format" in str(e) or "json" in str(e).lower()):
                logger.warning("JSON mode not supported, retrying without it.")
                kwargs.pop("response_format", None)
                response = self._nvidia_client.chat.completions.create(**kwargs)
            else:
                raise

        msg = response.choices[0].message
        if msg.content:
            return msg.content
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            return reasoning
        raise RuntimeError(
            f"NVIDIA NIM returned empty response from {self.nvidia_cfg.model_name}."
        )

    def _generate_gemini(self, prompt: str) -> str:
        """Generate text using the Google Gemini API (fallback)."""
        if not self._gemini_client:
            raise RuntimeError("Gemini client not initialized (missing API key).")

        response = self._gemini_client.models.generate_content(
            model=self.gemini_cfg.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_MESSAGE,
                temperature=self.gemini_cfg.temperature,
                max_output_tokens=self.gemini_cfg.max_output_tokens,
                top_p=self.gemini_cfg.top_p,
            ),
        )
        return response.text

    def _generate(self, prompt: str, force_json: bool = False) -> tuple[str, str]:
        """
        Generate text using Gemini as primary, NVIDIA NIM as fallback.
        Returns: (generated_text, provider_used)
        """
        # ── Try Gemini first (better at synthesis, all question types) ────────
        if self._gemini_client:
            try:
                text = self._generate_gemini(prompt)
                return text, PROVIDER_GEMINI
            except Exception as e:
                logger.warning(
                    f"Gemini generation failed ({type(e).__name__}: {e}). "
                    "Falling back to NVIDIA NIM…"
                )

        # ── Fall back to NVIDIA NIM ───────────────────────────────────────────
        if self._nvidia_client:
            try:
                text = self._generate_nvidia(prompt, force_json=force_json)
                return text, PROVIDER_NVIDIA
            except Exception as e:
                logger.warning(
                    f"NVIDIA NIM generation also failed ({type(e).__name__}: {e})."
                )

        raise RuntimeError(
            "All LLM providers unavailable. "
            "Set NVIDIA_API_KEY and/or GEMINI_API_KEY in your .env file."
        )

    # ── Pipeline steps ───────────────────────────────────────────────────────

    def rewrite_query(self, question: str) -> tuple[str, str]:
        """
        Use the LLM to rewrite the user's question into an optimized
        search query for better retrieval.

        Returns:
            (rewritten_query, provider_used)
        """
        try:
            prompt = QUERY_REWRITE_PROMPT.format(question=question)
            rewritten, provider = self._generate(prompt)
            rewritten = rewritten.strip()
            logger.info(
                f"[{provider}] Query rewritten: '{question}' → '{rewritten}'"
            )
            return rewritten, provider
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
            return question, "none"

    def retrieve_context(self, query: str, original_question: str) -> list:
        """
        Retrieve relevant chunks using both the rewritten query
        and the original question, then deduplicate.
        """
        results_rewritten = self.vector_store.search(query)
        results_original  = self.vector_store.search(original_question)

        seen = {}
        for chunk in results_rewritten + results_original:
            cid = chunk["chunk_id"]
            if cid not in seen or chunk["score"] > seen[cid]["score"]:
                seen[cid] = chunk

        merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return merged[: config.retrieval.top_k]

    def format_context(self, chunks: list) -> str:
        """Format retrieved chunks into a context string for the prompt."""
        if not chunks:
            return "[No relevant context found]"

        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = (
                f"--- Chunk {i} | Document: {chunk['document']} | "
                f"Page: {chunk['page']} | Relevance: {chunk['score']:.3f} ---"
            )
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n".join(parts)

    def detect_comparison(self, question: str) -> bool:
        """Detect if the question is asking for a comparison."""
        keywords = [
            "compare", "comparison", "versus", "vs", "vs.",
            "difference between", "differences between",
            "better", "which one", "side by side", "side-by-side",
        ]
        q = question.lower()
        return any(kw in q for kw in keywords)

    def generate_answer(
        self,
        question: str,
        context: str,
        is_comparison: bool = False,
    ) -> dict:
        """
        Generate a grounded answer with the assembled context.

        Tries NVIDIA NIM first; auto-falls back to Gemini on any error.

        Returns:
            Parsed JSON response dict with keys:
            answerable, answer, citations, confidence, llm_provider
        """
        template = COMPARISON_PROMPT if is_comparison else QA_PROMPT
        prompt   = template.format(context=context, question=question)

        last_error  = None
        provider_used = "none"
        max_attempts  = max(self.nvidia_cfg.max_retries, self.gemini_cfg.max_retries) + 1

        for attempt in range(max_attempts):
            try:
                raw_text, provider_used = self._generate(prompt, force_json=True)
                raw_text = raw_text.strip()

                parsed = self._parse_json_response(raw_text)
                self._validate_response(parsed)

                parsed["llm_provider"] = provider_used
                logger.info(
                    f"[{provider_used}] Answer generated (attempt {attempt + 1}): "
                    f"answerable={parsed.get('answerable')}, "
                    f"confidence={parsed.get('confidence')}"
                )
                return parsed

            except Exception as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    prompt += (
                        "\n\nIMPORTANT: Your previous response was not valid JSON. "
                        "Please respond with ONLY a valid JSON object, no markdown."
                    )

        logger.error(f"All generation attempts failed: {last_error}")
        return {
            "answerable":    False,
            "answer":        (
                "I encountered an error generating the response. "
                "Please try rephrasing your question."
            ),
            "citations":     [],
            "confidence":    "low",
            "llm_provider":  provider_used,
            "error":         str(last_error),
        }

    # ── JSON helpers ─────────────────────────────────────────────────────────

    def _parse_json_response(self, raw_text: str) -> dict:
        """Extract and parse JSON from the LLM's response.
        Handles models that embed literal newlines inside JSON strings.
        """
        # ── Attempt 1: direct parse ──────────────────────────────────────────
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # ── Attempt 2: extract from markdown code fence ──────────────────────
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # ── Attempt 3: extract brace block ───────────────────────────────────
        brace_start = raw_text.find("{")
        brace_end   = raw_text.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            candidate = raw_text[brace_start:brace_end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

            # ── Attempt 4: sanitize unescaped newlines inside string values ──
            # Replace literal newlines that appear inside JSON strings with \n
            sanitized = re.sub(
                r'("(?:[^"\\]|\\.)*")',
                lambda m: m.group(0).replace("\n", "\\n").replace("\r", ""),
                candidate,
                flags=re.DOTALL,
            )
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {raw_text[:200]}...")

    @staticmethod
    def _validate_response(parsed: dict):
        """Validate that the parsed response has the required fields."""
        for field_name in ("answerable", "answer", "citations", "confidence"):
            if field_name not in parsed:
                raise ValueError(f"Missing required field: {field_name}")

        if not isinstance(parsed["answerable"], bool):
            if isinstance(parsed["answerable"], str):
                parsed["answerable"] = parsed["answerable"].lower() == "true"
            else:
                raise ValueError(
                    f"'answerable' must be boolean, got {type(parsed['answerable'])}"
                )

        if parsed["confidence"] not in ("high", "medium", "low"):
            parsed["confidence"] = "medium"

    # ── Public API ───────────────────────────────────────────────────────────

    def answer_question(self, question: str) -> dict:
        """
        Full pipeline: rewrite → retrieve → generate → validate.

        Args:
            question: The user's natural-language question.

        Returns:
            Response dict with answer, citations, confidence, llm_provider,
            and pipeline metadata.
        """
        # Step 1: Rewrite query
        rewritten_query, rewrite_provider = self.rewrite_query(question)

        # Step 2: Retrieve relevant chunks
        chunks = self.retrieve_context(rewritten_query, question)

        # Step 3: Format context
        context = self.format_context(chunks)

        # Step 4: Detect comparison
        is_comparison = self.detect_comparison(question)

        # Step 5: Generate answer (NVIDIA → Gemini fallback)
        result = self.generate_answer(question, context, is_comparison)

        # Step 6: Attach pipeline metadata
        result["rewritten_query"]      = rewritten_query
        result["num_chunks_retrieved"] = len(chunks)
        result["chunks"] = [
            {
                "document": c["document"],
                "page":     c["page"],
                "score":    round(c["score"], 3),
                "preview":  c["text"][:150] + "…" if len(c["text"]) > 150 else c["text"],
            }
            for c in chunks
        ]
        return result
