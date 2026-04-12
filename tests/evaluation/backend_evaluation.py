from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# Path setup that keeps the project root importable when this file is run directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Environment flags that keep model libraries quieter during evaluation runs.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

try:
    # Logging setup that suppresses extra Hugging Face noise in the console output.
    from huggingface_hub import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    pass

try:
    # Logging setup that does the same for transformers warnings.
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    pass

from backend.app.services.chatbot_service import ChatbotService
from backend.app.services.intent_service import IntentService
from backend.app.services.retriever import TfidfRetriever

TESTS_DIR = ROOT / "tests"
DATASETS_DIR = TESTS_DIR / "datasets"
RESULTS_DIR = TESTS_DIR / "results"


def load_cases(filename: str) -> Any:
    # Dataset loader that keeps the evaluation inputs in one place so the report can be regenerated easily.
    path = DATASETS_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_intent_service() -> IntentService:
    # Intent service builder that loads the model once for reuse across the evaluation functions.
    service = IntentService()
    service.load()
    return service


def build_retriever() -> TfidfRetriever:
    # Retriever builder that points to the saved local index rather than rebuilding anything here.
    retriever = TfidfRetriever(
        index_dir=str(ROOT / "backend" / "index"),
        chunks_path=str(ROOT / "data" / "chunks" / "chunks.jsonl"),
    )
    retriever.load()
    return retriever


def build_chatbot_service(
    intent_service: IntentService | None = None,
    retriever: TfidfRetriever | None = None,
) -> ChatbotService:
    # Chatbot service builder that reuses the loaded services to keep the evaluation pass quicker.
    service = ChatbotService()
    if intent_service is not None:
        service.intent = intent_service
    if retriever is not None:
        service.retriever = retriever
    return service


def _round_score(value: Any) -> float | None:
    # Formatting helper that keeps the JSON report easier to read.
    if value is None:
        return None
    return round(float(value), 3)


def _average(values: list[float]) -> float:
    # Small helper used when reporting average confidence values.
    return sum(values) / len(values) if values else 0.0


def evaluate_intent_classification(intent_service: IntentService, cases: list[dict[str, Any]]) -> dict[str, Any]:
    # Evaluation routine that measures the classifier on the labelled intent dataset.
    rows = []
    correct = 0
    confidences = []
    correct_confidences = []
    incorrect_confidences = []

    for case in cases:
        query = case["query"]
        expected = case["expected_label"]
        # Per-case evaluation that keeps the predicted label and confidence for later reporting.
        label, confidence = intent_service.predict(query)
        matched = label == expected
        confidence_value = float(confidence)
        correct += int(matched)
        confidences.append(confidence_value)
        if matched:
            correct_confidences.append(confidence_value)
        else:
            incorrect_confidences.append(confidence_value)
        rows.append(
            {
                "query": query,
                "expected_label": expected,
                "predicted_label": label,
                "confidence": _round_score(confidence_value),
                "matched": matched,
            }
        )

    total = len(rows)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "average_confidence": _round_score(_average(confidences)),
        "average_confidence_correct": _round_score(_average(correct_confidences)),
        "average_confidence_incorrect": _round_score(_average(incorrect_confidences)) if incorrect_confidences else None,
        "min_confidence": _round_score(min(confidences)) if confidences else None,
        "max_confidence": _round_score(max(confidences)) if confidences else None,
        "rows": rows,
    }


def _first_hit_rank(results: list[dict[str, Any]], expected_url_contains: str, top_k: int) -> int:
    # Rank helper that returns the first position where the expected page appears in the results.
    expected = expected_url_contains.lower()
    for rank, result in enumerate(results[:top_k], start=1):
        url = (result.get("url") or "").lower()
        if expected in url:
            return rank
    return 0


def evaluate_retrieval_performance(retriever: TfidfRetriever, cases: list[dict[str, Any]]) -> dict[str, Any]:
    # Evaluation routine that scores retrieval with simple ranking metrics.
    rows = []
    acc1 = 0
    acc3 = 0
    reciprocal_rank_sum = 0.0

    for case in cases:
        query = case["query"]
        expected = case["expected_url_contains"]
        # Top-k setting that matches the retrieval metrics reported in the dissertation.
        results = retriever.search(query, top_k=3)
        # Retrieval row that stores the returned URLs as well as the first correct rank.
        rank = _first_hit_rank(results, expected, top_k=3)
        acc1 += int(rank == 1)
        acc3 += int(rank > 0)
        reciprocal_rank_sum += 1.0 / rank if rank else 0.0
        rows.append(
            {
                "group": case.get("group", ""),
                "query": query,
                "expected_url_contains": expected,
                "first_hit_rank": rank,
                "top_urls": [result.get("url") for result in results[:3]],
            }
        )

    total = len(rows)
    return {
        "total": total,
        "accuracy_at_1": acc1 / total if total else 0.0,
        "accuracy_at_3": acc3 / total if total else 0.0,
        "mrr_at_3": reciprocal_rank_sum / total if total else 0.0,
        "rows": rows,
    }


def evaluate_navigation_accuracy(chatbot_service: ChatbotService, cases: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    # Evaluation routine that checks both successful navigation and false page opens.
    positive_rows = []
    negative_rows = []
    positive_correct = 0
    false_positives = 0

    for index, case in enumerate(cases.get("positive", [])):
        session_id = f"nav-positive-{index}"
        # Positive case that should trigger a page open to the expected URL.
        _, _, actions, _, _, _, _, _, _, _ = chatbot_service.respond(session_id, case["query"])
        actual_url = actions[0].payload["url"] if actions else ""
        matched = actual_url == case["expected_url"]
        positive_correct += int(matched)
        positive_rows.append(
            {
                "query": case["query"],
                "expected_url": case["expected_url"],
                "actual_url": actual_url,
                "matched": matched,
            }
        )

    for index, case in enumerate(cases.get("negative", [])):
        session_id = f"nav-negative-{index}"
        # Negative case that should not trigger page opening at all.
        _, _, actions, _, _, _, _, _, _, _ = chatbot_service.respond(session_id, case["query"])
        opened_page = bool(actions)
        false_positives += int(opened_page)
        negative_rows.append(
            {
                "query": case["query"],
                "opened_page": opened_page,
            }
        )

    positive_total = len(positive_rows)
    negative_total = len(negative_rows)
    return {
        "positive_total": positive_total,
        "positive_accuracy": positive_correct / positive_total if positive_total else 0.0,
        "negative_total": negative_total,
        "false_positive_rate": false_positives / negative_total if negative_total else 0.0,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
    }


def infer_backend_route(chatbot_service: ChatbotService, session_id: str, reply: str, actions: list[Any]) -> str:
    # Route inference step that reads the final reply and saved state instead of duplicating service logic.
    state = chatbot_service.session_state.get(session_id, {})
    if reply.startswith("Summarising,"):
        return "summary"
    if actions:
        return "navigation"
    source = state.get("last_source") or ""
    if source == "fixed":
        return "intent"
    if source == "kb":
        return "kb"
    if source == "retrieval":
        return "retrieval"
    if source == "clarification":
        return "clarification"
    return source or "unknown"


def evaluate_backend_routing(chatbot_service: ChatbotService, cases: list[dict[str, Any]]) -> dict[str, Any]:
    # Evaluation routine that checks whether each query takes the expected backend path.
    rows = []
    correct = 0

    for index, case in enumerate(cases):
        session_id = f"route-{index}"
        # End-to-end case that checks which backend path the final response actually used.
        reply, _, actions, _, _, _, _, _, _, _ = chatbot_service.respond(session_id, case["query"])
        actual_route = infer_backend_route(chatbot_service, session_id, reply, actions)
        matched = actual_route == case["expected_route"]
        correct += int(matched)
        rows.append(
            {
                "query": case["query"],
                "expected_route": case["expected_route"],
                "actual_route": actual_route,
                "matched": matched,
            }
        )

    total = len(rows)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "rows": rows,
    }


def review_page_summaries(chatbot_service: ChatbotService, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Review helper that collects the actual summary outputs for qualitative discussion in the report.
    rows = []
    marker = "\nWould you like me to open that page for you?"

    for index, case in enumerate(cases):
        session_id = f"summary-review-{index}"
        # Summary review case that keeps both the matched URL and the returned summary text.
        reply, citations, _, quick, _, _, _, _, _, _ = chatbot_service.respond(session_id, case["query"])
        summary_text = reply.split(marker, 1)[0] if marker in reply else reply
        actual_url = citations[0].url if citations else ""
        rows.append(
            {
                "query": case["query"],
                "expected_url": case["expected_url"],
                "actual_url": actual_url,
                "matched_expected_url": actual_url == case["expected_url"],
                "summary": summary_text,
                "open_confirmation_present": marker in reply,
                "quick_replies": quick,
            }
        )

    return rows


def run_all_evaluations() -> dict[str, Any]:
    # Evaluation runner that produces the full JSON report in one pass.
    intent_service = build_intent_service()
    retriever = build_retriever()
    chatbot_service = build_chatbot_service(intent_service=intent_service, retriever=retriever)

    return {
        # Report sections that line up with the evaluation categories used in the dissertation.
        "score_definitions": {
            "intent_confidence": "Classifier confidence returned by the intent model for a predicted label.",
        },
        "intent_classification_performance": evaluate_intent_classification(
            intent_service,
            load_cases("intent_dataset.json"),
        ),
        "retrieval_performance": evaluate_retrieval_performance(
            retriever,
            load_cases("retrieval_dataset.json"),
        ),
        "navigation_accuracy": evaluate_navigation_accuracy(
            chatbot_service,
            load_cases("navigation_dataset.json"),
        ),
        "end_to_end_backend_routing_performance": evaluate_backend_routing(
            chatbot_service,
            load_cases("routing_dataset.json"),
        ),
        "page_summarisation_qualitative_review": review_page_summaries(
            chatbot_service,
            load_cases("summary_review_dataset.json"),
        ),
    }


def main() -> None:
    # Main entry point that writes the report straight into tests/results/report.json.
    report = run_all_evaluations()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "report.json"

    with report_path.open("w", encoding="utf-8") as handle:
        # Output file that stores the full evaluation results in a reusable JSON format.
        json.dump(report, handle, ensure_ascii=False, indent=2)

    intent_results = report["intent_classification_performance"]
    retrieval_results = report["retrieval_performance"]
    navigation_results = report["navigation_accuracy"]
    routing_results = report["end_to_end_backend_routing_performance"]
    summary_review = report["page_summarisation_qualitative_review"]

    print("Intent classification performance")
    print(f"  Accuracy: {intent_results['correct']}/{intent_results['total']} = {intent_results['accuracy']:.3f}")
    print(f"  Average confidence: {intent_results['average_confidence']:.3f}")

    print("\nRetrieval performance")
    print(f"  Accuracy@1: {retrieval_results['accuracy_at_1']:.3f}")
    print(f"  Accuracy@3: {retrieval_results['accuracy_at_3']:.3f}")
    print(f"  MRR@3: {retrieval_results['mrr_at_3']:.3f}")

    print("\nNavigation accuracy")
    print(f"  Positive accuracy: {navigation_results['positive_accuracy']:.3f}")
    print(f"  False-positive rate: {navigation_results['false_positive_rate']:.3f}")

    print("\nEnd-to-end backend routing performance")
    print(f"  Accuracy: {routing_results['correct']}/{routing_results['total']} = {routing_results['accuracy']:.3f}")

    print("\nShort qualitative review of page summarisation")
    for item in summary_review:
        print(f"  Query: {item['query']}")
        print(f"  URL matched: {item['actual_url']}")
        print(f"  Summary: {item['summary']}")

    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
