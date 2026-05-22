"""
embeddings/sbert_engine.py
==========================
Local SBERT embedding engine — NO external API calls.
Primary: sentence-transformers all-MiniLM-L6-v2 (384-dim, CPU)
Fallback: TF-IDF with a fixed vocabulary size (consistent across calls)
"""

import numpy as np
from typing import List

try:
    from sentence_transformers import SentenceTransformer as _ST
    _SBERT = _ST("all-MiniLM-L6-v2")
    HAS_SBERT = True
except Exception:
    _SBERT    = None
    HAS_SBERT = False

try:
    from sklearn.feature_extraction.text import HashingVectorizer
    # HashingVectorizer has a FIXED output dimension — safe for cross-call use
    _HASHER   = HashingVectorizer(n_features=512, norm="l2", alternate_sign=False)
    HAS_TFIDF = True
except ImportError:
    _HASHER   = None
    HAS_TFIDF = False


class EmbeddingEngine:
    """
    Unified embedding interface.
    SBERT (primary): consistent 384-dim vectors, works across separate encode_documents calls.
    HashingVectorizer (fallback): fixed 512-dim vectors, stateless — always consistent.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if HAS_SBERT and _SBERT is not None:
            self.model   = _SBERT
            self.dim     = _SBERT.get_sentence_embedding_dimension()
            self.backend = "sbert"
        elif HAS_TFIDF:
            self.model   = None
            self.dim     = 512
            self.backend = "hashing-tfidf"
        else:
            raise RuntimeError(
                "No embedding backend available. "
                "Run: pip install sentence-transformers  (or scikit-learn as fallback)"
            )
        print(f"[EmbeddingEngine] Backend: {self.backend}  dim={self.dim}")

    def encode_documents(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts → L2-normalised float32 vectors, shape (N, dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.backend == "sbert":
            vecs = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 200,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vecs.astype(np.float32)

        # HashingVectorizer fallback — stateless, fixed 512-dim output always
        mat = _HASHER.transform(texts).toarray().astype(np.float32)
        # Already L2-normalised by HashingVectorizer(norm='l2')
        return mat

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]
