"""
retrieval/faiss_retriever.py
============================
Production-grade FAISS ANN retrieval for plagiarism detection.

Contains:
  - FAISSRetriever     : document-level ANN search with persistence
  - FastSpanRetriever  : sentence-level top-K span matching (O(n log n))
  - StreamingEmbedder  : memory-efficient batch embedding generator
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Iterator

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

try:
    from sklearn.preprocessing import normalize as _sk_normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────
# Document-level ANN retriever  (unchanged interface)
# ─────────────────────────────────────────────────────────────

class FAISSRetriever:
    """
    Document-level FAISS index.
    Supports add / search / persist / restore.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.doc_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None  # fallback only

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None
            print("[FAISSRetriever] faiss not found -- using brute-force cosine fallback.")

    def reset(self):
        """
        Wipes all in-memory FAISS index and doc_ids completely.
        """
        self.doc_ids = []
        self._embeddings = None
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = None


    def add_embeddings(self, doc_ids: List[str], embeddings: np.ndarray):
        if len(embeddings) == 0:
            return
        embeddings = embeddings.astype(np.float32)
        if HAS_FAISS:
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            self._embeddings = (
                np.vstack([self._embeddings, embeddings])
                if self._embeddings is not None else embeddings
            )
        self.doc_ids.extend(doc_ids)

    def search_candidates(
        self,
        query: np.ndarray,
        k: int = 10,
        threshold: float = 0.75,
    ) -> List[Dict]:
        if not self.doc_ids:
            return []
        k = min(k, len(self.doc_ids))
        query = query.astype(np.float32).reshape(1, -1)

        if HAS_FAISS:
            faiss.normalize_L2(query)
            D, I = self.index.search(query, k)
            return [
                {"doc_id": self.doc_ids[idx], "similarity": float(d)}
                for d, idx in zip(D[0], I[0])
                if idx >= 0 and float(d) >= threshold
            ]
        else:
            if self._embeddings is None:
                return []
            sims = _cos_sim(query, self._embeddings)[0]
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
            return [
                {"doc_id": self.doc_ids[i], "similarity": float(s)}
                for i, s in ranked if float(s) >= threshold
            ]

    def save(self, path: str):
        """Persist FAISS index + doc_ids to disk for instant reload."""
        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, path + ".faiss")
            with open(path + ".meta", "wb") as f:
                pickle.dump({"doc_ids": self.doc_ids, "dimension": self.dimension}, f)
            print(f"[FAISSRetriever] Saved index -> {path}.faiss ({len(self.doc_ids)} docs)")

    def load(self, path: str) -> bool:
        """Restore FAISS index from disk. Returns True on success."""
        if not HAS_FAISS:
            return False
        faiss_path = path + ".faiss"
        meta_path  = path + ".meta"
        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            return False
        try:
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.doc_ids  = meta["doc_ids"]
            self.dimension = meta.get("dimension", self.dimension)
            print(f"[FAISSRetriever] Loaded index <- {faiss_path} ({len(self.doc_ids)} docs)")
            return True
        except Exception as e:
            print(f"[FAISSRetriever] Load failed: {e}")
            return False


# ─────────────────────────────────────────────────────────────
# Sentence-level FAISS span retriever
# ─────────────────────────────────────────────────────────────

class FastSpanRetriever:
    """
    Production-grade sentence-level ANN span detector.

    Replaces O(n^2) nested loops with FAISS top-K search.
    Complexity: approx O(n log n) — scales to large documents.

    Usage:
        retriever = FastSpanRetriever(embedding_engine, top_k=5)
        results   = retriever.retrieve(source_sentences, target_sentences)
    """

    def __init__(self, embedding_engine, top_k: int = 5, threshold: float = 0.80):
        self.emb       = embedding_engine
        self.top_k     = top_k
        self.threshold = threshold

    def _build_index(self, sentences: List[str]):
        """Encode sentences and return (faiss_index, embeddings)."""
        embeddings = self.emb.encode_documents(sentences, batch_size=32)
        embeddings = embeddings.astype(np.float32)

        if HAS_SKLEARN:
            embeddings = _sk_normalize(embeddings)
        elif HAS_FAISS:
            faiss.normalize_L2(embeddings)

        if HAS_FAISS:
            dim   = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            return index, embeddings
        else:
            return None, embeddings

    def retrieve(
        self,
        source_sentences: List[str],
        target_sentences: List[str],
    ) -> List[Dict]:
        """
        Find top-K similar sentence pairs between source and target.

        Returns list of dicts:
            {source_sentence, target_sentence, similarity}
        Filtered to pairs with similarity >= self.threshold.
        """
        if not source_sentences or not target_sentences:
            return []

        src_index, src_embeddings = self._build_index(source_sentences)
        tgt_embeddings = self.emb.encode_documents(target_sentences, batch_size=32).astype(np.float32)

        if HAS_SKLEARN:
            tgt_embeddings = _sk_normalize(tgt_embeddings)
        elif HAS_FAISS:
            faiss.normalize_L2(tgt_embeddings)

        k = min(self.top_k, len(source_sentences))
        results = []

        if HAS_FAISS and src_index is not None:
            D, I = src_index.search(tgt_embeddings, k)
            for tgt_idx in range(len(target_sentences)):
                for rank in range(k):
                    src_idx    = int(I[tgt_idx][rank])
                    similarity = float(D[tgt_idx][rank])
                    if src_idx >= 0 and similarity >= self.threshold:
                        results.append({
                            "source_sentence": source_sentences[src_idx],
                            "target_sentence": target_sentences[tgt_idx],
                            "similarity":      round(similarity, 4),
                        })
        else:
            # Fallback: brute-force cosine
            sims = _cos_sim(tgt_embeddings, src_embeddings)
            for tgt_idx in range(len(target_sentences)):
                ranked = np.argsort(sims[tgt_idx])[::-1][:k]
                for src_idx in ranked:
                    similarity = float(sims[tgt_idx][src_idx])
                    if similarity >= self.threshold:
                        results.append({
                            "source_sentence": source_sentences[int(src_idx)],
                            "target_sentence": target_sentences[tgt_idx],
                            "similarity":      round(similarity, 4),
                        })

        # Deduplicate and sort by similarity descending
        seen = set()
        deduped = []
        for r in sorted(results, key=lambda x: x["similarity"], reverse=True):
            key = (r["source_sentence"][:60], r["target_sentence"][:60])
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return deduped


# ─────────────────────────────────────────────────────────────
# Streaming batch embedder
# ─────────────────────────────────────────────────────────────

class StreamingEmbedder:
    """
    Memory-efficient embedding generator.

    Yields numpy arrays in batches instead of loading all embeddings
    into RAM at once. Prevents OOM crashes on large corpora.

    Usage:
        streamer = StreamingEmbedder(sbert_model, batch_size=64)
        for batch_embeddings in streamer.embed_documents(all_texts):
            faiss_index.add(batch_embeddings)
    """

    def __init__(self, model, batch_size: int = 64):
        self.model      = model
        self.batch_size = batch_size

    def embed_documents(self, documents: List[str]) -> Iterator[np.ndarray]:
        """Yield float32 embedding batches, L2-normalised."""
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            try:
                embeddings = self.model.encode(
                    batch,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                yield np.array(embeddings, dtype=np.float32)
            except Exception as e:
                print(f"[StreamingEmbedder] Batch {i//self.batch_size} failed: {e}")
                continue

    def embed_all(self, documents: List[str]) -> np.ndarray:
        """Convenience: collect all batches into a single array."""
        parts = list(self.embed_documents(documents))
        if not parts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.vstack(parts)
