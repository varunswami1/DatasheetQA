"""
DatasheetQA — Evaluation Module

Measures QA pipeline accuracy against a ground-truth question set.
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of a single evaluation question."""
    question: str
    expected_keywords: list[str]
    actual_answer: str
    confidence: str
    provider: str
    latency_seconds: float
    keyword_hit_rate: float = 0.0
    passed: bool = False


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    total_questions: int = 0
    passed: int = 0
    failed: int = 0
    avg_latency: float = 0.0
    avg_keyword_hit_rate: float = 0.0
    provider_counts: dict = field(default_factory=dict)
    results: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total_questions if self.total_questions > 0 else 0.0


# Ground-truth Q&A pairs for evaluation
EVAL_QUESTIONS = [
    {
        "question": "What is the maximum system voltage?",
        "keywords": ["1500", "voltage", "dc"],
    },
    {
        "question": "What is the temperature coefficient of Pmax?",
        "keywords": ["-0.34", "pmax", "temperature"],
    },
    {
        "question": "What are the mechanical dimensions of the module?",
        "keywords": ["mm", "dimension", "length", "width"],
    },
    {
        "question": "What is the module efficiency?",
        "keywords": ["efficiency", "%", "20"],
    },
    {
        "question": "What warranty is provided?",
        "keywords": ["warranty", "year", "power"],
    },
]


def evaluate_pipeline(pipeline, threshold: float = 0.5) -> EvalReport:
    """
    Run evaluation against the ground-truth question set.

    Args:
        pipeline: RAGPipeline instance with indexed documents.
        threshold: Keyword hit rate threshold to consider a question "passed".

    Returns:
        EvalReport with per-question results and summary statistics.
    """
    report = EvalReport()
    latencies = []
    hit_rates = []

    for qa in EVAL_QUESTIONS:
        question = qa["question"]
        keywords = [k.lower() for k in qa["keywords"]]

        start = time.time()
        try:
            result = pipeline.answer_question(question)
        except Exception as e:
            logger.error(f"Pipeline error on '{question}': {e}")
            result = {"answer": "", "confidence": "low", "llm_provider": "error"}
        elapsed = time.time() - start

        answer = result.get("answer", "").lower()
        hits = sum(1 for kw in keywords if kw in answer)
        hit_rate = hits / len(keywords) if keywords else 0.0
        passed = hit_rate >= threshold

        eval_result = EvalResult(
            question=question,
            expected_keywords=qa["keywords"],
            actual_answer=result.get("answer", "")[:200],
            confidence=result.get("confidence", "unknown"),
            provider=result.get("llm_provider", "unknown"),
            latency_seconds=round(elapsed, 2),
            keyword_hit_rate=round(hit_rate, 2),
            passed=passed,
        )
        report.results.append(asdict(eval_result))
        latencies.append(elapsed)
        hit_rates.append(hit_rate)

        provider = result.get("llm_provider", "unknown")
        report.provider_counts[provider] = report.provider_counts.get(provider, 0) + 1

        if passed:
            report.passed += 1
        else:
            report.failed += 1

        logger.info(
            f"{'✅' if passed else '❌'} [{provider}] ({elapsed:.1f}s) "
            f"hit_rate={hit_rate:.0%} | Q: {question[:60]}"
        )

    report.total_questions = len(EVAL_QUESTIONS)
    report.avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0
    report.avg_keyword_hit_rate = round(sum(hit_rates) / len(hit_rates), 2) if hit_rates else 0

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from config import config
    from vector_store import VectorStore
    from rag_pipeline import RAGPipeline

    vs = VectorStore()
    if not vs.load(config.index_folder, "datasheet_index"):
        print("❌ No indexed documents. Upload a PDF first via the web app.")
        exit(1)

    pipeline = RAGPipeline(vs)
    print(f"\nRunning evaluation on {len(EVAL_QUESTIONS)} questions...\n")

    report = evaluate_pipeline(pipeline)

    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT")
    print(f"{'='*50}")
    print(f"Accuracy       : {report.accuracy:.0%} ({report.passed}/{report.total_questions})")
    print(f"Avg Latency    : {report.avg_latency:.1f}s")
    print(f"Avg Keyword Hit: {report.avg_keyword_hit_rate:.0%}")
    print(f"Providers used : {report.provider_counts}")
    print(f"{'='*50}\n")

    with open("eval_report.json", "w") as f:
        json.dump(report.__dict__, f, indent=2)
    print("Report saved to eval_report.json")
