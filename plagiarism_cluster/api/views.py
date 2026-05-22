"""
api/views.py
============
All API endpoints — 100% local processing, NO external API calls.
FIXED VERSION
"""

import io
import os
import json
import logging
import hashlib
import sys

from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

logger = logging.getLogger(__name__)

# ============================================================================
# Add project root to sys.path
# ============================================================================

_BASE = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

# ============================================================================
# Lazy pipeline singleton
# ============================================================================

_PIPELINE = None


def get_pipeline():
    global _PIPELINE

    if _PIPELINE is None:
        from pipeline_core.pipeline import PlagiarismPipeline

        _PIPELINE = PlagiarismPipeline()

    return _PIPELINE


# ============================================================================
# Text Extraction Utility
# ============================================================================

def extract_text(file_obj, filename: str) -> str:
    """
    Extract text from:
        - PDF
        - DOCX
        - TXT
    """

    ext = filename.rsplit(".", 1)[-1].lower()

    # ------------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------------

    if ext == "pdf":

        try:
            import pdfplumber

            with pdfplumber.open(
                io.BytesIO(file_obj.read())
            ) as pdf:

                text = "\n".join(
                    p.extract_text() or ""
                    for p in pdf.pages
                )

                return text.strip()

        except ImportError:
            raise ValueError(
                "pdfplumber not installed"
            )

    # ------------------------------------------------------------------------
    # DOCX
    # ------------------------------------------------------------------------

    if ext == "docx":

        try:
            import mammoth

            result = mammoth.extract_raw_text(
                io.BytesIO(file_obj.read())
            )

            return result.value.strip()

        except ImportError:
            raise ValueError(
                "mammoth not installed"
            )

    # ------------------------------------------------------------------------
    # TXT / fallback
    # ------------------------------------------------------------------------

    raw = file_obj.read()

    for enc in ("utf-8", "latin-1", "cp1252"):

        try:
            return raw.decode(enc)

        except UnicodeDecodeError:
            continue

    return raw.decode(
        "utf-8",
        errors="replace"
    )


# ============================================================================
# GET /api/health/
# ============================================================================

def health(request):

    return JsonResponse({
        "status": "ok",
        "api": "plagiarism-detector"
    })


# ============================================================================
# POST /api/similarity/
# ============================================================================

@method_decorator(csrf_exempt, name="dispatch")
class SimilarityView(APIView):

    parser_classes = (
        MultiPartParser,
        FormParser
    )

    def post(self, request, *args, **kwargs):

        doc_a = request.FILES.get("doc_a")
        doc_b = request.FILES.get("doc_b")

        if not doc_a or not doc_b:

            return Response({
                "error":
                    "Two files required: doc_a and doc_b"
            }, status=400)

        try:

            text_a = extract_text(
                doc_a,
                doc_a.name
            )

            text_b = extract_text(
                doc_b,
                doc_b.name
            )

        except Exception as e:

            logger.exception("Extraction failed")

            return Response({
                "error":
                    f"File extraction failed: {e}"
            }, status=400)

        # --------------------------------------------------------------------
        # Validate text size
        # --------------------------------------------------------------------

        if len(text_a.split()) < 10:

            return Response({
                "error":
                    "Document A too short"
            }, status=400)

        if len(text_b.split()) < 10:

            return Response({
                "error":
                    "Document B too short"
            }, status=400)

        # --------------------------------------------------------------------
        # Run pipeline
        # --------------------------------------------------------------------

        try:

            result = get_pipeline().analyse_two_documents(
                doc_a.name,
                text_a,
                doc_b.name,
                text_b
            )

            return Response(result)

        except Exception as e:

            logger.exception(
                "Similarity analysis failed"
            )

            return Response({
                "error": str(e)
            }, status=500)


# ============================================================================
# POST /api/upload/
# ============================================================================

@method_decorator(csrf_exempt, name="dispatch")
class BatchUploadView(APIView):

    parser_classes = (
        MultiPartParser,
        FormParser
    )

    def post(self, request, *args, **kwargs):

        files = request.FILES.getlist("documents")

        if not files:

            return Response({
                "error": "No files provided"
            }, status=400)

        docs = {}

        # --------------------------------------------------------------------
        # Read all uploaded files
        # --------------------------------------------------------------------

        for idx, f in enumerate(files):

            try:

                doc_id = (
                    f"{f.name}_{idx}"
                    if f.name
                    else f"document_{idx}"
                )

                fname = (
                    f.name
                    if f.name and "." in f.name
                    else f"document_{idx}.txt"
                )

                text = extract_text(f, fname)

                if len(text.split()) < 5:
                    continue

                docs[doc_id] = text

            except Exception as e:

                logger.exception(
                    f"Failed reading {getattr(f, 'name', 'file')}"
                )

                return Response({
                    "error":
                        f"Error reading "
                        f"{getattr(f, 'name', 'file')}: {e}"
                }, status=400)

        # --------------------------------------------------------------------
        # Empty corpus check
        # --------------------------------------------------------------------

        if not docs:

            return Response({
                "error":
                    "No valid documents found"
            }, status=400)

        # --------------------------------------------------------------------
        # Run pipeline
        # --------------------------------------------------------------------

        try:

            unique_docs, clusters, pairwise_similarity = (
                get_pipeline().process_corpus(docs)
            )

            return Response({

                "message":
                    "Processing complete",

                "processed":
                    len(unique_docs),

                "clusters":
                    len(clusters),

                "cluster_details": {
                    str(k): v
                    for k, v in clusters.items()
                },

                "pairwise_similarity":
                    pairwise_similarity,
            })

        except Exception as e:

            logger.exception(
                "Batch processing failed"
            )

            return Response({
                "error": str(e)
            }, status=500)


# ============================================================================
# POST /api/ai-detect/
# ============================================================================

@method_decorator(csrf_exempt, name="dispatch")
class AIDetectView(View):

    def post(self, request, *args, **kwargs):

        try:
            body = json.loads(request.body)

        except json.JSONDecodeError:

            return JsonResponse({
                "error": "Invalid JSON"
            }, status=400)

        text = body.get("text", "").strip()

        if len(text.split()) < 10:

            return JsonResponse({
                "error": "Text too short"
            }, status=400)

        try:

            from detection.ai_detector import (
                analyse_text_for_ai
            )

            result = analyse_text_for_ai(text)

            return JsonResponse(result)

        except Exception as e:

            logger.exception(
                "AI detection failed"
            )

            return JsonResponse({
                "error": str(e)
            }, status=500)


# ============================================================================
# POST /api/dataset-run/
# ============================================================================

@method_decorator(csrf_exempt, name="dispatch")
class DatasetRunView(View):

    def post(self, request, *args, **kwargs):

        from django.conf import settings as dj_settings

        data_dir = dj_settings.DATASET_DIR

        # --------------------------------------------------------------------
        # Validate dataset directory
        # --------------------------------------------------------------------

        if not os.path.isdir(data_dir):

            return JsonResponse({
                "error":
                    f"DATASET_DIR not found: {data_dir}"
            }, status=400)

        # --------------------------------------------------------------------
        # Parse request body
        # --------------------------------------------------------------------

        try:

            body = (
                json.loads(request.body)
                if request.body
                else {}
            )

        except Exception:

            body = {}

        output_dir = body.get("output_dir")

        print("[DATASET RUN]")
        print("DATASET_DIR:", data_dir)

        # --------------------------------------------------------------------
        # Run pipeline
        # --------------------------------------------------------------------

        try:

            summary = get_pipeline().run_on_dataset(
                data_dir,
                output_dir=output_dir
            )

            return JsonResponse(summary)

        except Exception as e:

            logger.exception(
                "Dataset run failed"
            )

            return JsonResponse({
                "error": str(e)
            }, status=500)


# ============================================================================
# POST /api/evaluate/
# ============================================================================

@method_decorator(csrf_exempt, name="dispatch")
class EvaluateView(View):

    def post(self, request, *args, **kwargs):

        from django.conf import settings as dj_settings

        # --------------------------------------------------------------------
        # Parse request
        # --------------------------------------------------------------------

        try:

            body = (
                json.loads(request.body)
                if request.body
                else {}
            )

        except Exception:

            body = {}

        dataset = body.get(
            "dataset",
            "pan2011"
        )

        task = body.get(
            "task",
            "external"
        )

        data_dir = dj_settings.DATASET_DIR

        # --------------------------------------------------------------------
        # Validate dataset path
        # --------------------------------------------------------------------

        if not os.path.isdir(data_dir):

            return JsonResponse({
                "error":
                    f"DATASET_DIR not found: {data_dir}"
            }, status=400)

        print("[EVALUATION]")
        print("Dataset:", dataset)
        print("Task:", task)
        print("DATASET_DIR:", data_dir)

        # --------------------------------------------------------------------
        # Run evaluation
        # --------------------------------------------------------------------

        try:

            metrics = get_pipeline().evaluate_against_gt(
                data_dir=data_dir,
                dataset=dataset,
                task=task
            )

            return JsonResponse(metrics)

        except Exception as e:

            logger.exception(
                "Evaluation failed"
            )

            return JsonResponse({
                "error": str(e)
            }, status=500)