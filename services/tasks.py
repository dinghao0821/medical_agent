"""Async task definitions (P2).

Heavy, non-interactive jobs that benefit from running off the request path:
RAG document ingestion (and, optionally, image segmentation).

Each job's real logic lives in a plain ``_function`` so it can be executed
synchronously when the queue is disabled. When Celery is available the same
functions are also registered as tasks. See ``services/task_queue.py`` for the
submit facade that chooses async vs sync.
"""

import logging

from services.celery_app import celery_app, CELERY_AVAILABLE

logger = logging.getLogger(__name__)


# ---- Plain implementations (used for the synchronous fallback) ----

def _ingest_directory(directory_path: str):
    """Ingest all documents in a directory into the RAG vector store."""
    from config import Config
    from agents.rag_agent import MedicalRAG

    logger.info("[task] Ingesting directory: %s", directory_path)
    return MedicalRAG(Config()).ingest_directory(directory_path)


def _ingest_file(document_path: str):
    """Ingest a single document into the RAG vector store."""
    from config import Config
    from agents.rag_agent import MedicalRAG

    logger.info("[task] Ingesting file: %s", document_path)
    return MedicalRAG(Config()).ingest_file(document_path)


def _segment_image(image_type: str, image_path: str, output_path: str):
    """Run image segmentation/classification off the request path.

    NOTE: not wired into the synchronous /upload flow by default; provided so the
    plan's "image inference async" has a landing spot for a future job-status API.
    """
    from config import Config
    from agents.image_analysis_agent import ImageAnalysisAgent

    agent = ImageAnalysisAgent(config=Config())
    if image_type == "brain_tumor":
        return agent.segment_brain_tumor(image_path, output_path)
    if image_type == "skin_lesion":
        return agent.segment_skin_lesion(image_path, output_path)
    if image_type == "chest_xray":
        return agent.classify_chest_xray(image_path)
    raise ValueError(f"Unknown image_type: {image_type}")


# ---- Celery task registrations (only when Celery is importable) ----

if CELERY_AVAILABLE and celery_app is not None:
    ingest_directory_task = celery_app.task(name="rag.ingest_directory")(_ingest_directory)
    ingest_file_task = celery_app.task(name="rag.ingest_file")(_ingest_file)
    segment_image_task = celery_app.task(name="cv.segment_image")(_segment_image)
else:  # pragma: no cover
    ingest_directory_task = None
    ingest_file_task = None
    segment_image_task = None
