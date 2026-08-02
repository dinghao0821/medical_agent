"""RAGAS evaluation harness for the Medical RAG pipeline (P4, optional).

Runs a small QA dataset through the existing RAG agent and scores it with RAGAS
(faithfulness / answer relevancy / context precision & recall). This is an
offline quality-regression tool, not part of the serving path.

Prerequisites:
    pip install ragas datasets
    - valid LLM/embedding API keys in .env (same as the app)
    - a populated Qdrant knowledge base (run ingest_rag_data.py first)

Usage:
    python -m evaluation.ragas_eval [path/to/eval_dataset.json]
"""

import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragas_eval")

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "eval_dataset.json")


def _load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_contexts(rag_response) -> list:
    """Best-effort extraction of retrieved context strings from a RAG response."""
    contexts = []
    sources = rag_response.get("sources") if isinstance(rag_response, dict) else None
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, dict):
                contexts.append(str(s.get("content") or s.get("text") or s))
            else:
                contexts.append(str(s))
    return contexts or ["(no retrieved context)"]


def run(dataset_path: str = DEFAULT_DATASET):
    # Lazy, guarded imports so the repo doesn't require ragas to be installed.
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
    except Exception as e:
        logger.error("RAGAS/datasets not installed (%s). Run: pip install ragas datasets", e)
        return 1

    from config import Config
    from agents.rag_agent import MedicalRAG

    config = Config()
    rag = MedicalRAG(config)

    items = _load_dataset(dataset_path)
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in items:
        q = item["question"]
        logger.info("Querying: %s", q)
        try:
            resp = rag.process_query(q)
            answer = resp.get("response", "") if isinstance(resp, dict) else str(resp)
            if not isinstance(answer, str):
                answer = getattr(answer, "content", str(answer))
            contexts = _collect_contexts(resp)
        except Exception as e:
            logger.warning("Query failed for '%s' (%s)", q, e)
            answer, contexts = "", ["(error)"]

        records["question"].append(q)
        records["answer"].append(answer)
        records["contexts"].append(contexts)
        records["ground_truth"].append(item.get("ground_truth", ""))

    dataset = Dataset.from_dict(records)
    logger.info("Running RAGAS evaluation on %d samples...", len(items))
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    print("\n=== RAGAS scores ===")
    print(result)
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    raise SystemExit(run(path))
