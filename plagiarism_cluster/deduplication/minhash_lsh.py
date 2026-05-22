"""
deduplication/minhash_lsh.py
=========================================
Advanced Research-Grade Deduplication Engine

Features
--------
1. Exact duplicate detection (SHA-256)
2. Near-duplicate detection (MinHash + LSH)
3. Semantic-ready preprocessing
4. Unicode normalization
5. Proper tokenization
6. Lemmatization (spaCy optional)
7. Shingling support
8. Similarity scoring
9. Persistent storage
10. O(1) exact hash lookup
11. Configurable thresholds
12. Safe fallback modes

Architecture
-------------
SHA256
   +
MinHash LSH
   +
Jaccard Verification
   +
(Ready for SBERT/FAISS extension)

Requirements
-------------
pip install datasketch spacy
python -m spacy download en_core_web_sm
"""

import os
import re
import pickle
import hashlib
import unicodedata

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ============================================================
# Optional Libraries
# ============================================================

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except Exception:
    HAS_DATASKETCH = False

try:
    # pyrefly: ignore [missing-import]
    import spacy

    NLP = spacy.load("en_core_web_sm")
    HAS_SPACY = True

except Exception:
    NLP = None
    HAS_SPACY = False


# ============================================================
# Utility Functions
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize unicode text.
    """
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    """

    text = normalize_unicode(text)

    text = text.lower()

    text = re.sub(r"http\\S+", " ", text)

    text = re.sub(r"[^a-z0-9\\s]", " ", text)

    text = re.sub(r"\\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Advanced tokenization with optional lemmatization.
    """

    text = clean_text(text)

    if HAS_SPACY:

        doc = NLP(text)

        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
        ]

        return tokens

    return text.split()


def generate_shingles(
    tokens: List[str],
    k: int = 3
) -> List[str]:
    """
    Generate k-shingles.
    """

    if len(tokens) < k:
        return tokens

    shingles = [
        " ".join(tokens[i:i+k])
        for i in range(len(tokens) - k + 1)
    ]

    return shingles


# ============================================================
# Main Deduplicator
# ============================================================

class Deduplicator:

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        shingle_size: int = 3,
        min_tokens: int = 20,
        storage_path: Optional[str] = None
    ):

        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.min_tokens = min_tokens
        self.storage_path = storage_path

        # Exact duplicate lookup
        self.hash_to_docs = defaultdict(list)

        # Store MinHashes
        self.minhashes = {}

        # Store cleaned tokens
        self.documents = {}

        # LSH
        if HAS_DATASKETCH:

            self.lsh = MinHashLSH(
                threshold=threshold,
                num_perm=num_perm
            )

        else:

            self.lsh = None

            print(
                "[Deduplicator] datasketch not installed. "
                "Using SHA-256 exact duplicate detection only."
            )

        # Load persisted index
        self.load()

    def reset(self):
        """
        Wipes all in-memory index states completely.
        """
        self.hash_to_docs = defaultdict(list)
        self.minhashes = {}
        self.documents = {}
        if HAS_DATASKETCH:
            self.lsh = MinHashLSH(
                threshold=self.threshold,
                num_perm=self.num_perm
            )
        else:
            self.lsh = None



    # ========================================================
    # SHA256
    # ========================================================

    def sha256(self, text: str) -> str:

        return hashlib.sha256(
            text.encode("utf-8", errors="replace")
        ).hexdigest()


    # ========================================================
    # MinHash
    # ========================================================

    def build_minhash(
        self,
        shingles: List[str]
    ) -> Optional[MinHash]:

        if not HAS_DATASKETCH:
            return None

        m = MinHash(num_perm=self.num_perm)

        for shingle in shingles:

            m.update(
                shingle.encode("utf-8")
            )

        return m


    # ========================================================
    # Jaccard Similarity
    # ========================================================

    @staticmethod
    def jaccard_similarity(
        set1,
        set2
    ) -> float:

        intersection = len(set1 & set2)

        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union


    # ========================================================
    # Main Duplicate Detection
    # ========================================================

    def is_duplicate(
        self,
        doc_id: str,
        text: str
    ) -> Tuple[bool, List[Dict]]:

        """
        Returns:
        (
            is_duplicate,
            [
                {
                    "doc_id": str,
                    "similarity": float,
                    "type": "exact" | "near"
                }
            ]
        )
        """

        # ----------------------------------------------------
        # Normalize + tokenize
        # ----------------------------------------------------

        tokens = tokenize(text)

        if len(tokens) < self.min_tokens:

            return False, []

        shingles = generate_shingles(
            tokens,
            self.shingle_size
        )

        shingle_set = set(shingles)

        # ----------------------------------------------------
        # Exact Duplicate Detection
        # ----------------------------------------------------

        content_hash = self.sha256(
            " ".join(tokens)
        )

        if content_hash in self.hash_to_docs:

            matches = [
                {
                    "doc_id": existing_id,
                    "similarity": 1.0,
                    "type": "exact"
                }
                for existing_id in self.hash_to_docs[content_hash]
            ]

            return True, matches

        # ----------------------------------------------------
        # Near Duplicate Detection
        # ----------------------------------------------------

        matches = []

        if self.lsh is not None:

            minhash = self.build_minhash(shingles)

            candidate_ids = self.lsh.query(minhash)

            for candidate_id in candidate_ids:

                candidate_shingles = self.documents.get(
                    candidate_id,
                    set()
                )

                similarity = self.jaccard_similarity(
                    shingle_set,
                    candidate_shingles
                )

                if similarity >= self.threshold:

                    matches.append(
                        {
                            "doc_id": candidate_id,
                            "similarity": round(similarity, 4),
                            "type": "near"
                        }
                    )

            # Insert into index
            try:

                self.lsh.insert(
                    doc_id,
                    minhash
                )

            except ValueError:
                pass

            self.minhashes[doc_id] = minhash

        # ----------------------------------------------------
        # Store Document
        # ----------------------------------------------------

        self.hash_to_docs[content_hash].append(doc_id)

        self.documents[doc_id] = shingle_set

        # ----------------------------------------------------
        # Save Index
        # ----------------------------------------------------

        self.save()

        return len(matches) > 0, matches


    # ========================================================
    # Persistence
    # ========================================================

    def save(self):
        if not self.storage_path:
            return

        data = {
            "hash_to_docs": dict(self.hash_to_docs),
            "documents": self.documents,
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "shingle_size": self.shingle_size
        }

        try:

            with open(self.storage_path, "wb") as f:

                pickle.dump(data, f)

        except Exception as e:

            print(f"[Deduplicator] Save failed: {e}")


    def load(self):
        if not self.storage_path or not os.path.exists(self.storage_path):
            return

        try:

            with open(self.storage_path, "rb") as f:

                data = pickle.load(f)

            self.hash_to_docs = defaultdict(
                list,
                data.get("hash_to_docs", {})
            )

            self.documents = data.get(
                "documents",
                {}
            )

            # Rebuild LSH
            if HAS_DATASKETCH:

                self.lsh = MinHashLSH(
                    threshold=self.threshold,
                    num_perm=self.num_perm
                )

                for doc_id, shingles in self.documents.items():

                    m = self.build_minhash(
                        list(shingles)
                    )

                    self.minhashes[doc_id] = m

                    try:

                        self.lsh.insert(doc_id, m)

                    except ValueError:
                        pass

        except Exception as e:

            print(f"[Deduplicator] Load failed: {e}")


    # ========================================================
    # Remove Document
    # ========================================================

    def remove_document(
        self,
        doc_id: str
    ):

        if doc_id in self.documents:
            del self.documents[doc_id]

        if doc_id in self.minhashes:
            del self.minhashes[doc_id]

        hashes_to_remove = []

        for h, ids in self.hash_to_docs.items():

            if doc_id in ids:

                ids.remove(doc_id)

            if not ids:
                hashes_to_remove.append(h)

        for h in hashes_to_remove:
            del self.hash_to_docs[h]

        self.save()


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    dedup = Deduplicator(
        threshold=0.8,
        num_perm=128,
        shingle_size=3
    )

    docs = {
        "doc1": """
        Artificial Intelligence is transforming industries
        through automation and intelligent systems.
        """,

        "doc2": """
        Artificial Intelligence transforms industries
        using automation and intelligent technologies.
        """,

        "doc3": """
        The solar system contains planets, moons,
        asteroids, and comets.
        """
    }

    for doc_id, text in docs.items():

        is_dup, matches = dedup.is_duplicate(
            doc_id,
            text
        )

        print("\\n", "=" * 50)

        print(f"Document: {doc_id}")

        print(f"Duplicate: {is_dup}")

        print("Matches:")

        for match in matches:

            print(
                f"  -> {match['doc_id']} "
                f"(similarity={match['similarity']}, "
                f"type={match['type']})"
            )