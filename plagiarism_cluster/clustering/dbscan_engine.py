"""
clustering/dbscan_engine.py
============================
Improved clustering engine with:

✓ HDBSCAN primary backend
✓ DBSCAN fallback
✓ Automatic all-noise recovery
✓ Better logging
✓ Safer clustering handling
✓ Small dataset protection
✓ Cluster statistics
✓ Robust pipeline support

HDBSCAN advantages over DBSCAN:
  - Variable density support
  - More robust on noisy embeddings
  - Better scaling
  - Soft clustering support

DBSCAN fallback:
  eps=0.2 ≈ cosine similarity threshold 0.80
"""

import numpy as np
from typing import List, Dict

try:
    # pyrefly: ignore [missing-import]
    import hdbscan as _hdbscan_lib

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

from sklearn.cluster import DBSCAN


class ClusterEngine:
    """
    Document clustering engine.

    Uses:
        1. HDBSCAN (preferred)
        2. DBSCAN fallback

    Features:
        - Automatic noise recovery
        - Better debugging
        - Stable clustering for small datasets
    """

    def __init__(
        self,
        eps: float = 0.2,
        min_samples: int = 2,
    ):
        self.eps = eps
        self.min_samples = min_samples

        if HAS_HDBSCAN:
            print(
                "[ClusterEngine] Backend: HDBSCAN "
                "(variable density, robust clustering)"
            )
        else:
            print(
                "[ClusterEngine] Backend: DBSCAN "
                "(install hdbscan for better clustering)"
            )

    def cluster_documents(
        self,
        embeddings: np.ndarray,
        doc_ids: List[str],
    ) -> Dict[int, List[str]]:
        """
        Cluster document embeddings.

        Args:
            embeddings : numpy embedding matrix
            doc_ids    : corresponding document ids

        Returns:
            Dict[int, List[str]]
        """

        # =========================
        # SAFETY CHECKS
        # =========================

        if len(embeddings) == 0:
            print("[ClusterEngine] No embeddings received")
            return {}

        if len(embeddings) != len(doc_ids):
            raise ValueError(
                "Embeddings count does not match doc_ids count"
            )

        # Single document
        if len(embeddings) == 1:
            print("[ClusterEngine] Single document only")
            return {0: [doc_ids[0]]}

        print(
            f"[ClusterEngine] Clustering "
            f"{len(embeddings)} documents"
        )

        # =========================
        # CLUSTERING
        # =========================

        try:
            # Small datasets -> DBSCAN often works better
            if len(embeddings) < 10:
                print("[ClusterEngine] Using DBSCAN (small dataset)")
                labels = self._run_dbscan(embeddings)

            elif HAS_HDBSCAN:
                print("[ClusterEngine] Using HDBSCAN")
                labels = self._run_hdbscan(embeddings)

            else:
                print("[ClusterEngine] Using DBSCAN fallback")
                labels = self._run_dbscan(embeddings)

        except Exception as e:
            print(f"[ClusterEngine] Clustering failed: {e}")

            # Emergency fallback
            labels = np.zeros(len(embeddings), dtype=int)

        # =========================
        # FIX: ALL DOCUMENTS NOISE
        # =========================

        unique_labels = set(labels)

        # If ALL docs are marked as noise (-1)
        if unique_labels == {-1}:
            print(
                "[ClusterEngine] WARNING: "
                "All documents marked as noise"
            )

            print(
                "[ClusterEngine] Recovery mode: "
                "Putting all documents into cluster 0"
            )

            labels = np.zeros(len(embeddings), dtype=int)

        # =========================
        # BUILD CLUSTERS
        # =========================

        clusters: Dict[int, List[str]] = {}

        for doc_id, label in zip(doc_ids, labels):

            # Skip noise
            if label == -1:
                continue

            clusters.setdefault(int(label), []).append(doc_id)

        # =========================
        # LOGGING
        # =========================

        print("[ClusterEngine] Labels:", labels)

        if not clusters:
            print("[ClusterEngine] No valid clusters created")

        for label, docs in clusters.items():
            print(
                f"[Cluster {label}] "
                f"({len(docs)} docs)"
            )

            for d in docs:
                print(f"   -> {d}")

        return clusters

    def _run_hdbscan(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        HDBSCAN clustering.

        Uses euclidean metric because:
        L2-normalised euclidean distance
        approximates cosine distance.
        """

        clusterer = _hdbscan_lib.HDBSCAN(
            min_cluster_size=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,
        )

        labels = clusterer.fit_predict(embeddings)

        return labels

    def _run_dbscan(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        DBSCAN fallback clustering.

        cosine metric:
            similarity-based clustering
        """

        db = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="cosine",
            n_jobs=-1,
        )

        labels = db.fit_predict(embeddings)

        return labels

    def get_backend(self) -> str:
        """
        Return active clustering backend.
        """

        return "hdbscan" if HAS_HDBSCAN else "dbscan"