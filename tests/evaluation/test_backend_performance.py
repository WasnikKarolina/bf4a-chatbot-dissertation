from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parent
# Path setup that lets pytest import the shared evaluation helpers from this folder.
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from backend_evaluation import (
    build_chatbot_service,
    build_intent_service,
    build_retriever,
    evaluate_backend_routing,
    evaluate_intent_classification,
    evaluate_navigation_accuracy,
    evaluate_retrieval_performance,
    load_cases,
)


@pytest.fixture(scope="session")
def intent_service():
    # Session fixture that keeps the test run from reloading the same files repeatedly.
    return build_intent_service()


@pytest.fixture(scope="session")
def retriever():
    # Session fixture that reuses the same loaded retriever across the evaluation tests.
    return build_retriever()


@pytest.fixture(scope="session")
def chatbot_service(intent_service, retriever):
    # Session fixture that builds one chatbot service using the shared intent and retrieval services.
    return build_chatbot_service(intent_service=intent_service, retriever=retriever)


def test_intent_classification_performance(intent_service):
    # Intent test that checks whether classifier performance stays above the chosen project threshold.
    results = evaluate_intent_classification(intent_service, load_cases("intent_dataset.json"))
    # Threshold that is high enough to catch regressions without making the test brittle.
    assert results["accuracy"] >= 0.90


def test_retrieval_performance(retriever):
    # Retrieval test that checks the main ranking metrics used in the report.
    results = evaluate_retrieval_performance(retriever, load_cases("retrieval_dataset.json"))
    assert results["accuracy_at_1"] >= 0.80
    assert results["accuracy_at_3"] >= 0.95
    assert results["mrr_at_3"] >= 0.85


def test_navigation_accuracy(chatbot_service):
    # Navigation test that checks both successful opens and zero false positives.
    results = evaluate_navigation_accuracy(chatbot_service, load_cases("navigation_dataset.json"))
    assert results["positive_accuracy"] == 1.0
    assert results["false_positive_rate"] == 0.0


def test_end_to_end_backend_routing_performance(chatbot_service):
    # Routing test that checks whether the backend still sends queries through the expected path.
    results = evaluate_backend_routing(chatbot_service, load_cases("routing_dataset.json"))
    assert results["accuracy"] >= 0.90
