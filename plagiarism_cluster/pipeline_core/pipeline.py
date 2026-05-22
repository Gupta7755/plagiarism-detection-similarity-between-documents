"""
pipeline_core/pipeline.py
==========================
Main orchestrator. All analysis is done locally -- no external APIs.

Scalability improvements:
  - FastSpanRetriever  : O(n log n) sentence-level FAISS ANN instead of O(n^2)
  - StreamingEmbedder  : batch generator to avoid RAM spikes on large corpora
  - HDBSCAN clustering : variable density, more robust than fixed-eps DBSCAN
  - Persistent FAISS   : index saved/restored to skip rebuild on warm starts
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from preprocessing.text_cleaner  import preprocess_document
from deduplication.minhash_lsh   import Deduplicator
from embeddings.sbert_engine     import EmbeddingEngine
from retrieval.faiss_retriever   import FAISSRetriever, FastSpanRetriever, StreamingEmbedder
from clustering.dbscan_engine    import ClusterEngine
from detection.span_detector     import SpanDetector
from detection.ai_detector       import analyse_text_for_ai
from evaluation.pan_evaluator    import PANOutputGenerator, Evaluator, AIDetectionEvaluator
from ingestion.data_loader       import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# Singleton embedding engine (loaded once, reused everywhere)
_ENGINE: Optional[EmbeddingEngine] = None

def get_engine() -> EmbeddingEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = EmbeddingEngine()
    return _ENGINE


class PlagiarismPipeline:
    def __init__(self, eps: float = 0.35, span_threshold: float = 0.80):
        self.engine           = get_engine()
        self.dedup            = Deduplicator()
        self.retriever        = FAISSRetriever(dimension=self.engine.dim)
        self.clusterer        = ClusterEngine(eps=eps)
        self.span_det         = SpanDetector(self.engine, similarity_threshold=span_threshold)
        self.fast_span        = FastSpanRetriever(self.engine, top_k=5, threshold=span_threshold)
        self.streamer         = StreamingEmbedder(self.engine.model if hasattr(self.engine, 'model') else None, batch_size=64)

    # ── process a batch of documents ─────────────────────────────────────────

    def process_corpus(self, documents: Dict[str, str]) -> Tuple[Dict, Dict, List]:
        """
        Preprocess → dedup → embed → FAISS → DBSCAN.
        Returns (unique_docs, clusters, pairwise_similarity).
        """
        self.dedup.reset()
        self.retriever.reset()

        unique_docs: Dict[str, dict] = {}
        for doc_id, text in documents.items():
            if not text or len(text.split()) < 10:
                continue
            is_dup, _ = self.dedup.is_duplicate(doc_id, text)
            if not is_dup:
                unique_docs[doc_id] = preprocess_document(text)

        if not unique_docs:
            return {}, {}, []

        doc_ids    = list(unique_docs.keys())
        embeddings = self._embed(doc_ids, unique_docs)

        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(embeddings)

        print("\n[DEBUG] Cosine Similarity Matrix:")
        print(sim_matrix)

        pairwise_results = self.compute_pairwise_similarity(
            embeddings,
            doc_ids
        )

        print("\n[DEBUG] Pairwise Similarity Results:")
        print(pairwise_results)

        self.retriever.add_embeddings(doc_ids, embeddings.copy())
        clusters = self.clusterer.cluster_documents(embeddings, doc_ids)
        return unique_docs, clusters, pairwise_results

    # ── span detection between two docs ──────────────────────────────────────

    def detect_pair(
        self,
        doc1_id: str, doc2_id: str,
        unique_docs: Dict[str, dict],
    ) -> List[Dict]:
        d1 = unique_docs.get(doc1_id, {})
        d2 = unique_docs.get(doc2_id, {})
        if not d1 or not d2:
            return []
        return self.span_det.detect_plagiarism(
            d1["clean_text"], d1["sentences"],
            d2["clean_text"], d2["sentences"],
        )

    # ── full two-document similarity API ─────────────────────────────────────

    def analyse_two_documents(
        self,
        name_a: str, text_a: str,
        name_b: str, text_b: str,
    ) -> Dict:
        """
        Complete similarity analysis of two documents.
        Used by the /api/similarity/ endpoint.
        """
        import hashlib
        from sklearn.metrics.pairwise import cosine_similarity as _cos

        pre_a = preprocess_document(text_a)
        pre_b = preprocess_document(text_b)

        clean_a, sents_a = pre_a["clean_text"], pre_a["sentences"]
        clean_b, sents_b = pre_b["clean_text"], pre_b["sentences"]

        # Exact duplicate
        hash_a = hashlib.sha256(clean_a.encode()).hexdigest()
        hash_b = hashlib.sha256(clean_b.encode()).hexdigest()
        exact  = hash_a == hash_b

        # Document-level embeddings
        emb_a  = self.engine.encode_single(clean_a[:5000])
        emb_b  = self.engine.encode_single(clean_b[:5000])
        cos    = float(np.dot(emb_a, emb_b))
        cos    = max(0.0, min(1.0, cos))

        # MinHash near-dup
        near_dup  = False
        mh_score  = 0.0
        try:
            from datasketch import MinHash
            def _mh(t):
                m = MinHash(num_perm=128)
                for w in t.lower().split():
                    m.update(w.encode())
                return m
            mh_a    = _mh(clean_a)
            mh_b    = _mh(clean_b)
            mh_score = mh_a.jaccard(mh_b)
            near_dup = mh_score >= 0.8
        except Exception:
            pass

        # Span detection
        sim_spans  = self.span_det.find_similar_spans(sents_a, sents_b, threshold=0.80)
        exact_spans = [s for s in sim_spans if s["similarity"] >= 0.95]
        para_spans  = [s for s in sim_spans if 0.80 <= s["similarity"] < 0.95]

        # AI detection (fully local)
        ai_a = analyse_text_for_ai(clean_a)
        ai_b = analyse_text_for_ai(clean_b)

        # Verdict
        pct = int(cos * 100)
        if exact:
            verdict = "Exact Duplicate"
        elif near_dup or pct >= 85:
            verdict = "High Similarity"
        elif pct >= 60:
            verdict = "Moderate Similarity"
        elif pct >= 30:
            verdict = "Low Similarity"
        else:
            verdict = "Unique"

        return {
            "doc_a": {"name": name_a, "words": pre_a["word_count"],
                      "sentences": len(sents_a), "language": pre_a["language"],
                      "hash": hash_a[:16]},
            "doc_b": {"name": name_b, "words": pre_b["word_count"],
                      "sentences": len(sents_b), "language": pre_b["language"],
                      "hash": hash_b[:16]},
            "similarity": {
                "overall_pct":     pct,
                "cosine_score":    round(cos, 4),
                "minhash_jaccard": round(mh_score, 4),
                "exact_duplicate": exact,
                "near_duplicate":  near_dup,
                "verdict":         verdict,
            },
            "spans": {
                "total_matching": len(sim_spans),
                "exact":       [{"doc_a": s["doc_a_sentence"], "doc_b": s["doc_b_sentence"],
                                  "score": s["similarity"]} for s in exact_spans[:8]],
                "paraphrase":  [{"doc_a": s["doc_a_sentence"], "doc_b": s["doc_b_sentence"],
                                  "score": s["similarity"]} for s in para_spans[:8]],
            },
            "ai_detection": {"doc_a": ai_a, "doc_b": ai_b},
            "pipeline_stages": {
                "preprocessing": f"A:{len(sents_a)} sents | B:{len(sents_b)} sents",
                "deduplication": f"SHA-256 match:{exact} | MinHash:{round(mh_score,3)}",
                "embedding":     self.engine.backend,
                "cosine":        round(cos, 4),
                "spans":         f"{len(sim_spans)} above 0.80 ({len(exact_spans)} exact, {len(para_spans)} para)",
            },
        }

    # ── dataset batch pipeline ────────────────────────────────────────────────

    def run_on_dataset(self, data_dir: str, output_dir: Optional[str] = None) -> Dict:
        loader  = DataLoader(data_dir)
        corpus  = loader.load_all()
        if not corpus:
            return {"error": "no_documents_found"}

        print(f"[Pipeline] Processing {len(corpus)} documents ...")
        unique_docs, clusters, _ = self.process_corpus(corpus)
        print(f"[Pipeline] {len(unique_docs)} unique -> {len(clusters)} clusters.")

        xml_count = 0
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for label, members in clusters.items():
                if len(members) < 2:
                    continue
                susp = members[0]
                for src in members[1:]:
                    spans = self.detect_pair(susp, src, unique_docs)
                    if spans:
                        xml_path = os.path.join(output_dir, f"{susp}_vs_{src}.xml")
                        PANOutputGenerator.generate_xml(susp, src, spans, xml_path)
                        xml_count += 1

        return {
            "total_input":    len(corpus),
            "unique_docs":    len(unique_docs),
            "clusters":       len(clusters),
            "cluster_sizes":  {str(k): len(v) for k, v in clusters.items()},
            "xml_written":    xml_count,
            "cluster_backend": self.clusterer.get_backend(),
            "embedding_backend": self.engine.backend,
        }

    def evaluate_against_gt(self, data_dir: str, dataset: str = "pan2011") -> Dict:
        loader = DataLoader(data_dir)
        corpus = loader.load_all()
        gt     = loader.load_ground_truth(dataset=dataset)
        if not gt:
            return {"error": "no_ground_truth"}

        unique_docs, _, _ = self.process_corpus(corpus)

        gt_by_doc: Dict[str, List[Dict]] = {}
        for ann in gt:
            gt_by_doc.setdefault(ann["suspicious_doc"], []).append(ann)

        predictions: Dict[str, List[Dict]] = {}
        for susp_id, anns in gt_by_doc.items():
            src_ids  = list({a["source_doc"] for a in anns})
            all_spans: List[Dict] = []
            for src_id in src_ids:
                import os

                sk = next(
                    (
                        k for k in unique_docs
                        if os.path.basename(k) == os.path.basename(src_id)
                    ),
                    None
                )
                
                pk = next(
                    (
                        k for k in unique_docs
                        if os.path.basename(k) == os.path.basename(susp_id)
                    ),
                    None
                )
                if pk and sk:
                    all_spans.extend(self.detect_pair(pk, sk, unique_docs))
            predictions[susp_id] = all_spans

        return Evaluator.evaluate_corpus(predictions, gt_by_doc)

    # ── private helpers ───────────────────────────────────────────────────────

    def _embed(self, doc_ids: List[str], unique_docs: Dict[str, dict]) -> np.ndarray:
        """
        Embed documents using streaming batches to avoid RAM spikes.
        Each document is represented by the mean of its sentence embeddings (capped at 50).
        """
        vecs = []
        # Collect one representative text per doc (first 50 sentences joined)
        texts = []
        for doc_id in doc_ids:
            sents = unique_docs[doc_id]["sentences"][:50]
            texts.append(" ".join(sents) if sents else "")

        # Stream in batches of 64 to avoid OOM on large corpora
        all_embs = []
        for batch_start in range(0, len(texts), 64):
            batch = texts[batch_start:batch_start + 64]
            batch_embs = self.engine.encode_documents(batch, batch_size=64)
            all_embs.append(batch_embs)

        if all_embs:
            return np.vstack(all_embs).astype(np.float32)

        return np.zeros((len(doc_ids), self.engine.dim), dtype=np.float32)

    def compute_pairwise_similarity(self, embeddings, doc_ids):
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)

        pairs = []

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                score = float(sim_matrix[i][j])

                pairs.append({
                    "doc_a": doc_ids[i],
                    "doc_b": doc_ids[j],
                    "cosine_similarity": round(score, 4),
                    "percentage": round(score * 100, 2),
                })

        pairs.sort(
            key=lambda x: x["cosine_similarity"],
            reverse=True
        )

        return pairs