"""
PDF processing module for DatasheetQA.

Handles PDF text extraction (including tables), page-aware chunking,
and metadata preservation.
"""

import os
import re
import logging
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import pdfplumber

from config import config

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extracts text and tables from PDF datasheets with page tracking."""

    def __init__(self):
        self.chunk_cfg = config.chunking

    def extract_text_pymupdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from each page using PyMuPDF.

        Returns:
            List of dicts: [{"page": int, "text": str}, ...]
        """
        pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    pages.append({
                        "page": page_num + 1,
                        "text": text.strip()
                    })
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            raise
        return pages

    def extract_tables_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """
        Extract tables from each page using pdfplumber and convert to markdown.

        Returns:
            List of dicts: [{"page": int, "table_md": str}, ...]
        """
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        md = self._table_to_markdown(table)
                        if md:
                            tables.append({
                                "page": page_num,
                                "table_md": md
                            })
        except Exception as e:
            logger.error(f"pdfplumber table extraction failed for {pdf_path}: {e}")
        return tables

    @staticmethod
    def _table_to_markdown(table: List[List]) -> Optional[str]:
        """Convert a 2D table array to a markdown table string."""
        if not table or len(table) < 2:
            return None

        # Clean cells
        cleaned = []
        for row in table:
            cleaned_row = []
            for cell in row:
                cell_text = str(cell).strip() if cell else ""
                cell_text = cell_text.replace("\n", " ").replace("|", "\\|")
                cleaned_row.append(cell_text)
            cleaned.append(cleaned_row)

        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in cleaned)
        for row in cleaned:
            while len(row) < max_cols:
                row.append("")

        # Build markdown
        header = "| " + " | ".join(cleaned[0]) + " |"
        separator = "| " + " | ".join(["---"] * max_cols) + " |"
        body_lines = []
        for row in cleaned[1:]:
            body_lines.append("| " + " | ".join(row) + " |")

        return "\n".join([header, separator] + body_lines)

    def process_pdf(self, pdf_path: str, filename: str) -> List[Dict]:
        """
        Full PDF processing pipeline: extract text + tables, then chunk.

        Returns:
            List of chunk dicts:
            [{"text": str, "page": int, "document": str, "chunk_id": int}, ...]
        """
        logger.info(f"Processing PDF: {filename}")

        # Extract text pages
        text_pages = self.extract_text_pymupdf(pdf_path)

        # Extract tables
        tables = self.extract_tables_pdfplumber(pdf_path)

        # Merge tables into their respective pages
        page_content = {}
        for p in text_pages:
            page_content[p["page"]] = p["text"]

        for t in tables:
            page_num = t["page"]
            table_section = f"\n\n[TABLE]\n{t['table_md']}\n[/TABLE]\n"
            if page_num in page_content:
                page_content[page_num] += table_section
            else:
                page_content[page_num] = table_section

        # Chunk each page's content
        chunks = []
        chunk_id = 0
        for page_num in sorted(page_content.keys()):
            content = page_content[page_num]
            page_chunks = self._chunk_text(content)
            for chunk_text in page_chunks:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "document": filename,
                    "chunk_id": chunk_id
                })
                chunk_id += 1

        logger.info(f"Generated {len(chunks)} chunks from {filename}")
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks, respecting sentence boundaries
        where possible.
        """
        # Determine chunk size based on whether content has tables
        has_table = "[TABLE]" in text
        max_size = (
            self.chunk_cfg.table_chunk_size if has_table
            else self.chunk_cfg.chunk_size
        )
        overlap = self.chunk_cfg.chunk_overlap
        min_size = self.chunk_cfg.min_chunk_size

        if len(text) <= max_size:
            return [text] if len(text) >= min_size else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_size

            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence-ending punctuation near the end
                best_break = -1
                search_start = max(start + max_size // 2, start)
                for match in re.finditer(r'[.!?\n]\s', text[search_start:end]):
                    best_break = search_start + match.end()

                if best_break > start:
                    end = best_break

            chunk = text[start:end].strip()
            if len(chunk) >= min_size:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks
