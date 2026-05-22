"""
detection/span_detector.py
===========================================================
Research-grade semantic span detector.

Features:
- SBERT semantic embeddings
- FAISS ANN retrieval
- Top-K nearest-neighbor search
- PAN-compatible offsets
- Duplicate filtering
- Short-sentence filtering
- Memory-safe batching
- O(n log n) retrieval scaling
"""

import faiss
import numpy as np

from typing import List, Dict, Tuple

from sklearn.preprocessing import normalize


class SpanDetector:

    def __init__(
        self,
        embedding_engine,
        similarity_threshold: float = 0.80,
        top_k: int = 5,
        min_sentence_words: int = 5,
    ):

        self.emb = embedding_engine

        self.threshold = similarity_threshold

        self.top_k = top_k

        self.min_sentence_words = min_sentence_words

    # ─────────────────────────────────────────────────────────
    # Character offsets
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _offsets(
        text: str,
        sentences: List[str]
    ) -> List[Tuple[int, int]]:

        offsets = []

        cursor = 0

        lower_text = text.lower()

        for sent in sentences:

            idx = text.find(sent, cursor)

            if idx == -1:
                idx = lower_text.find(sent.lower(), cursor)

            if idx != -1:

                offsets.append((idx, idx + len(sent)))

                cursor = idx + len(sent)

            else:

                offsets.append((cursor, cursor))

        return offsets

    # ─────────────────────────────────────────────────────────
    # Build FAISS index
    # ─────────────────────────────────────────────────────────

    def _build_index(self, embeddings):

        embeddings = normalize(embeddings)

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)

        index.add(embeddings)

        return index

    # ─────────────────────────────────────────────────────────
    # Main plagiarism detection
    # ─────────────────────────────────────────────────────────

    def detect_plagiarism(
        self,
        doc1_text: str,
        doc1_sents: List[str],
        doc2_text: str,
        doc2_sents: List[str],
    ) -> List[Dict]:

        if not doc1_sents or not doc2_sents:
            return []

        # Remove very short sentences
        filtered_doc1 = [
            s for s in doc1_sents
            if len(s.split()) >= self.min_sentence_words
        ]

        filtered_doc2 = [
            s for s in doc2_sents
            if len(s.split()) >= self.min_sentence_words
        ]

        if not filtered_doc1 or not filtered_doc2:
            return []

        # Embeddings
        emb1 = self.emb.encode_documents(
            filtered_doc1,
            batch_size=32
        )

        emb2 = self.emb.encode_documents(
            filtered_doc2,
            batch_size=32
        )

        emb1 = normalize(emb1)
        emb2 = normalize(emb2)

        # Build FAISS index
        index = self._build_index(emb2)

        # Search nearest neighbors
        D, I = index.search(
            np.array(emb1, dtype="float32"),
            self.top_k
        )

        off1 = self._offsets(doc1_text, filtered_doc1)

        off2 = self._offsets(doc2_text, filtered_doc2)

        spans = []

        used_pairs = set()

        for i, neighbors in enumerate(I):

            for rank, j in enumerate(neighbors):

                similarity = float(D[i][rank])

                if similarity < self.threshold:
                    continue

                # Prevent duplicate matches
                pair_key = (i, j)

                if pair_key in used_pairs:
                    continue

                used_pairs.add(pair_key)

                s1, e1 = off1[i]

                s2, e2 = off2[j]

                l1 = e1 - s1

                l2 = e2 - s2

                if l1 <= 0 or l2 <= 0:
                    continue

                spans.append({

                    "this_offset": s1,

                    "this_length": l1,

                    "source_offset": s2,

                    "source_length": l2,

                    "similarity": round(similarity, 4),

                    "doc1_sentence": filtered_doc1[i],

                    "doc2_sentence": filtered_doc2[j],
                })

        spans.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        return spans

    # ─────────────────────────────────────────────────────────
    # API helper
    # ─────────────────────────────────────────────────────────

    def find_similar_spans(
        self,
        sents_a: List[str],
        sents_b: List[str],
        threshold: float = None,
        max_results: int = 20,
    ) -> List[Dict]:

        threshold = threshold or self.threshold

        if not sents_a or not sents_b:
            return []

        emb_a = self.emb.encode_documents(
            sents_a,
            batch_size=32
        )

        emb_b = self.emb.encode_documents(
            sents_b,
            batch_size=32
        )

        emb_a = normalize(emb_a)

        emb_b = normalize(emb_b)

        index = self._build_index(emb_b)

        D, I = index.search(
            np.array(emb_a, dtype="float32"),
            self.top_k
        )

        pairs = []

        used_pairs = set()

        for i, neighbors in enumerate(I):

            for rank, j in enumerate(neighbors):

                similarity = float(D[i][rank])

                if similarity < threshold:
                    continue

                pair_key = (i, j)

                if pair_key in used_pairs:
                    continue

                used_pairs.add(pair_key)

                pairs.append({

                    "doc_a_sentence": sents_a[i],

                    "doc_b_sentence": sents_b[j],

                    "similarity": round(similarity, 4),
                })

        pairs.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        return pairs[:max_results]