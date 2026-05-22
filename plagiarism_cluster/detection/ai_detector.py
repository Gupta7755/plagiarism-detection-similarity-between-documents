"""
detection/ai_detector.py
========================
Research-Grade Multi-Signal AI Detection Pipeline.

Signals used:
  1. Transformer classifier  (openai-community/roberta-base-openai-detector)
  2. GPT-2 perplexity        (low perplexity → likely AI)
  3. Stylometric analysis    (burstiness + sentence variance)
  4. Lexical diversity       (repetitive vocab → likely AI)
  5. Semantic uniformity     (SBERT cosine consistency → likely AI)
  6. Repetition analysis     (sentence-opening repeats → likely AI)
  7. Advanced Heuristic Engine  ← UPGRADED (see HeuristicEngine below)
       • Perplexity heuristic
       • Burstiness analysis
       • Vocabulary richness (TTR + hapax ratio)
       • N-gram repetition (bigrams, trigrams, sentence starters)
       • Formal tone detection (passive voice, transition words, academic vocab)
       • Human imperfection signals (contractions, hedges, personal markers)

All models run fully LOCAL — no external API calls.

Scoring Weights (Advanced Heuristic sub-scores, max 100):
  Perplexity heuristic    30
  Burstiness              20
  Repetition              15
  Formal tone             15
  Vocabulary pattern      10
  Human signals           10
  Total                  100

Final pipeline fusion:
  Transformer        50 %
  Heuristic engine   20 %   ← was 15 %, expanded to absorb richer signal
  Stylometric        15 %   ← was 20 %, slight reduction
  Perplexity (GPT2)  10 %
  Semantic + repeat   5 %
"""

import re
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter

# ─────────────────────────────────────────────────────────────
# Optional NLTK helpers (used by HeuristicEngine)
# ─────────────────────────────────────────────────────────────

try:
    import nltk
    # Silently ensure required data is present
    for _pkg in ("punkt", "averaged_perceptron_tagger", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{_pkg}")
        except LookupError:
            try:
                nltk.download(_pkg, quiet=True)
            except Exception:
                pass
    from nltk.tokenize import word_tokenize
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False


# ─────────────────────────────────────────────────────────────
# 1. Transformer classifier
# ─────────────────────────────────────────────────────────────

try:
    from transformers import pipeline as hf_pipeline

    _CLASSIFIER = hf_pipeline(
        "text-classification",
        model="openai-community/roberta-base-openai-detector",
        device=-1,
    )
    HAS_CLASSIFIER = True

except Exception:
    _CLASSIFIER = None
    HAS_CLASSIFIER = False


# ─────────────────────────────────────────────────────────────
# 2. GPT-2 perplexity model
# ─────────────────────────────────────────────────────────────

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    _PPL_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    _PPL_MODEL = GPT2LMHeadModel.from_pretrained("gpt2")
    _PPL_MODEL.eval()
    HAS_PERPLEXITY = True

except Exception:
    _PPL_TOKENIZER = None
    _PPL_MODEL = None
    HAS_PERPLEXITY = False


# ─────────────────────────────────────────────────────────────
# Utility: sentence splitter
# ─────────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ─────────────────────────────────────────────────────────────
# Utility: chunking for long documents
# ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1200) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# ─────────────────────────────────────────────────────────────
# Utility: tokenize words
# ─────────────────────────────────────────────────────────────

def tokenize_words(text: str) -> List[str]:
    if HAS_NLTK:
        try:
            return word_tokenize(text.lower())
        except Exception:
            pass
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


# ═════════════════════════════════════════════════════════════
#  Advanced Heuristic Engine
# ═════════════════════════════════════════════════════════════

class HeuristicEngine:
    """
    Explainable heuristic AI-detection engine.

    Produces a weighted score 0-100 and a list of human-readable
    reasons explaining each contributing signal.

    Sub-scores (each capped at their declared max):
        Perplexity heuristic  → 30 pts
        Burstiness            → 20 pts
        Repetition            → 15 pts
        Formal tone           → 15 pts
        Vocabulary pattern    → 10 pts
        Human signals         → 10 pts  (deduction: less AI)
        Total possible        100 pts
    """

    # ── Transition / formal words ──────────────────────────────
    _TRANSITION_WORDS = {
        "furthermore", "moreover", "additionally", "consequently",
        "therefore", "thus", "hence", "nevertheless", "nonetheless",
        "subsequently", "correspondingly", "accordingly", "in conclusion",
        "in summary", "to summarize", "to conclude", "in essence",
        "as a result", "for instance", "for example", "in contrast",
        "on the other hand", "it is worth noting", "it is important to note",
        "it should be noted", "it is essential", "it is crucial",
        "this demonstrates", "this highlights", "this underscores",
    }

    _ACADEMIC_VOCAB = {
        "utilize", "facilitate", "leverage", "implement", "enhance",
        "optimize", "demonstrate", "articulate", "proliferate",
        "comprehensive", "multifaceted", "paradigm", "framework",
        "methodology", "perspective", "context", "implications",
        "significant", "substantial", "fundamental", "inherent",
        "robust", "delve", "crucial", "vital", "pivotal",
        "shed light", "in the realm", "landscape", "ecosystem",
        "holistic", "synergy", "nuanced", "intricate", "encompasses",
    }

    _AI_HALLMARK_PHRASES = [
        r"\bin conclusion\b",
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\bdelve\b",
        r"\bcomprehensive(?:ly)?\b",
        r"\bultimately\b",
        r"\bfacilitate[sd]?\b",
        r"\beverage[sd]?\b",
        r"\brobust\b",
        r"\bin the realm of\b",
        r"\bsheds? light\b",
        r"\bit is (?:important|worth|essential|crucial) to (?:note|mention|highlight)\b",
        r"\bthis (?:paper|article|essay|study|document) (?:explores?|examines?|discusses?)\b",
        r"\bthe (?:purpose|aim|goal|objective) of this\b",
        r"\bit is (?:well[- ]known|widely acknowledged|generally accepted)\b",
        r"\bin today's (?:world|society|era|landscape)\b",
        r"\bone (?:must|should|cannot) (?:consider|acknowledge|recognize|underestimate)\b",
    ]
    _AI_RE = [re.compile(p, re.IGNORECASE) for p in _AI_HALLMARK_PHRASES]

    # ── Human imperfection signals ─────────────────────────────
    _CONTRACTIONS = re.compile(
        r"\b(?:don't|doesn't|can't|won't|isn't|aren't|wasn't|weren't|"
        r"didn't|hadn't|hasn't|haven't|shouldn't|wouldn't|couldn't|"
        r"i'm|i've|i'll|i'd|you're|you've|you'll|you'd|"
        r"we're|we've|we'll|we'd|they're|they've|they'll|they'd|"
        r"he's|she's|it's|that's|there's|here's|who's|what's)\b",
        re.IGNORECASE,
    )
    _HEDGE_WORDS = re.compile(
        r"\b(?:maybe|perhaps|probably|possibly|i think|i believe|i feel|"
        r"i guess|i suppose|i wonder|honestly|frankly|personally|"
        r"in my opinion|from my perspective|it seems|it appears|"
        r"i struggled|i found|i noticed|i realized|i remember|"
        r"i'm not sure|i wasn't sure|to be honest|to be fair)\b",
        re.IGNORECASE,
    )
    _EMOTIONAL_MARKERS = re.compile(
        r"\b(?:amazing|awful|terrible|wonderful|fantastic|horrible|"
        r"frustrating|exciting|boring|annoying|love|hate|excited|"
        r"nervous|worried|happy|sad|angry|disappointed|surprised|"
        r"shocked|grateful|sorry|unfortunately|luckily|thankfully)\b",
        re.IGNORECASE,
    )
    _INFORMAL_MARKERS = re.compile(
        r"\b(?:yeah|yep|nope|ok|okay|gonna|wanna|kinda|sorta|gotta|"
        r"btw|tbh|imo|lol|haha|oh well|anyway|anyways|so yeah|"
        r"like i said|as i said|you know|kind of|sort of|a bit|"
        r"a lot of|lots of)\b",
        re.IGNORECASE,
    )

    # ── Passive voice pattern ──────────────────────────────────
    _PASSIVE = re.compile(
        r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b",
        re.IGNORECASE,
    )

    # ─────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────

    def analyse(self, text: str, perplexity: float = 999.0) -> Dict:
        """
        Run the full heuristic pipeline.

        Args:
            text:        Input text to analyse.
            perplexity:  GPT-2 perplexity already computed by the outer
                         pipeline (avoids re-running the model).
                         Pass 999.0 if unavailable.

        Returns dict:
            probability   : 0-100 combined heuristic AI score
            reasons       : list[str] — human-readable explanation bullets
            sub_scores    : dict with each feature's raw score
            features      : dict with raw feature measurements
        """
        sentences = split_sentences(text)
        words = tokenize_words(text)

        reasons: List[str] = []
        sub_scores: Dict[str, int] = {}

        # ── 1. Perplexity heuristic ──────────────────────────
        ppl_score, ppl_reasons = self._perplexity_heuristic(perplexity)
        sub_scores["perplexity"] = ppl_score
        reasons.extend(ppl_reasons)

        # ── 2. Burstiness ────────────────────────────────────
        burst_score, burst_reasons, burst_feats = self._burstiness_heuristic(sentences)
        sub_scores["burstiness"] = burst_score
        reasons.extend(burst_reasons)

        # ── 3. Vocabulary richness ───────────────────────────
        vocab_score, vocab_reasons, vocab_feats = self._vocabulary_heuristic(words)
        sub_scores["vocabulary"] = vocab_score
        reasons.extend(vocab_reasons)

        # ── 4. Repetition detection ──────────────────────────
        rep_score, rep_reasons, rep_feats = self._repetition_heuristic(text, sentences, words)
        sub_scores["repetition"] = rep_score
        reasons.extend(rep_reasons)

        # ── 5. Formal tone ───────────────────────────────────
        tone_score, tone_reasons, tone_feats = self._formal_tone_heuristic(text, sentences, words)
        sub_scores["formal_tone"] = tone_score
        reasons.extend(tone_reasons)

        # ── 6. Human imperfection signals ───────────────────
        human_deduct, human_reasons, human_feats = self._human_signal_heuristic(text, words)
        sub_scores["human_signals"] = human_deduct  # negative: subtracts from AI score
        reasons.extend(human_reasons)

        # ── Weighted sum ─────────────────────────────────────
        raw = (
            ppl_score        # max 30
            + burst_score    # max 20
            + rep_score      # max 15
            + tone_score     # max 15
            + vocab_score    # max 10
            - human_deduct   # max 10 deduction
        )
        probability = int(min(100, max(0, raw)))

        features = {
            "perplexity": round(perplexity, 2) if perplexity != 999.0 else None,
            **burst_feats,
            **vocab_feats,
            **rep_feats,
            **tone_feats,
            **human_feats,
        }

        return {
            "probability": probability,
            "reasons": reasons,
            "sub_scores": sub_scores,
            "features": features,
        }

    # ─────────────────────────────────────────────────────────
    # 1. Perplexity heuristic (max 30 pts)
    # ─────────────────────────────────────────────────────────

    def _perplexity_heuristic(self, perplexity: float) -> Tuple[int, List[str]]:
        reasons = []
        if perplexity == 999.0:
            return 0, []  # model unavailable — contribute nothing

        if perplexity < 20:
            score = 30
            reasons.append(f"✔ Very low perplexity ({perplexity:.1f}) — text is highly predictable, strongly AI-like")
        elif perplexity < 30:
            score = 22
            reasons.append(f"✔ Low perplexity ({perplexity:.1f}) — text flows unusually smoothly")
        elif perplexity < 40:
            score = 12
            reasons.append(f"✔ Moderately low perplexity ({perplexity:.1f}) — slightly more predictable than typical human text")
        elif perplexity < 60:
            score = 4
            reasons.append(f"– Perplexity in normal range ({perplexity:.1f})")
        else:
            score = 0
            reasons.append(f"✘ High perplexity ({perplexity:.1f}) — text has human-like randomness")

        return score, reasons

    # ─────────────────────────────────────────────────────────
    # 2. Burstiness heuristic (max 20 pts)
    # ─────────────────────────────────────────────────────────

    def _burstiness_heuristic(
        self, sentences: List[str]
    ) -> Tuple[int, List[str], Dict]:
        reasons = []
        if len(sentences) < 3:
            return 0, [], {"sentence_variance": None, "burstiness": None, "avg_sent_len": None}

        lengths = [len(s.split()) for s in sentences]
        mean_len = float(np.mean(lengths))
        variance = float(np.var(lengths))
        std = float(np.std(lengths))
        burst = std / mean_len if mean_len > 0 else 0.0

        score = 0

        if variance < 30:
            score += 12
            reasons.append(f"✔ Very uniform sentence lengths (variance={variance:.1f}) — characteristic of AI output")
        elif variance < 60:
            score += 6
            reasons.append(f"✔ Low sentence length variance ({variance:.1f}) — AI texts tend to be more uniform")

        if burst < 0.35:
            score += 8
            reasons.append(f"✔ Very low burstiness ({burst:.3f}) — sentence rhythm lacks human irregularity")
        elif burst < 0.50:
            score += 4
            reasons.append(f"✔ Low burstiness ({burst:.3f}) — moderately uniform rhythm")
        elif burst > 0.80:
            reasons.append(f"✘ High burstiness ({burst:.3f}) — varied rhythm suggests human writing")

        score = min(score, 20)
        feats = {
            "sentence_variance": round(variance, 2),
            "burstiness": round(burst, 4),
            "avg_sent_len": round(mean_len, 1),
        }
        return score, reasons, feats

    # ─────────────────────────────────────────────────────────
    # 3. Vocabulary richness (max 10 pts)
    # ─────────────────────────────────────────────────────────

    def _vocabulary_heuristic(
        self, words: List[str]
    ) -> Tuple[int, List[str], Dict]:
        reasons = []
        alpha_words = [w for w in words if w.isalpha()]
        if not alpha_words:
            return 0, [], {"ttr": None, "hapax_ratio": None, "vocab_size": 0}

        total = len(alpha_words)
        unique = len(set(alpha_words))
        ttr = unique / total  # Type-Token Ratio

        # Hapax legomena: words that appear exactly once
        freq = Counter(alpha_words)
        hapax = sum(1 for v in freq.values() if v == 1)
        hapax_ratio = hapax / unique if unique > 0 else 0.0

        score = 0

        # AI text tends toward balanced, repetitive vocabulary (moderate TTR)
        # Human text: either very high TTR (rich vocab) or low TTR (casual/repetitive)
        if 0.45 <= ttr <= 0.65:
            score += 5
            reasons.append(f"✔ Vocabulary TTR ({ttr:.3f}) in the 'AI-typical' mid-range — balanced, non-extreme diversity")
        elif ttr < 0.40:
            score += 3
            reasons.append(f"✔ Low vocabulary diversity (TTR={ttr:.3f}) — repetitive word usage")
        else:
            reasons.append(f"✘ High vocabulary diversity (TTR={ttr:.3f}) — richer than typical AI output")

        # Low hapax ratio means AI reuses words without unique one-offs
        if hapax_ratio < 0.45:
            score += 5
            reasons.append(f"✔ Low hapax ratio ({hapax_ratio:.3f}) — fewer unique one-off words than typical human text")
        elif hapax_ratio > 0.65:
            reasons.append(f"✘ High hapax ratio ({hapax_ratio:.3f}) — many unique words typical of human writing")

        score = min(score, 10)
        feats = {
            "ttr": round(ttr, 4),
            "hapax_ratio": round(hapax_ratio, 4),
            "vocab_size": unique,
        }
        return score, reasons, feats

    # ─────────────────────────────────────────────────────────
    # 4. Repetition detection (max 15 pts)
    # ─────────────────────────────────────────────────────────

    def _repetition_heuristic(
        self, text: str, sentences: List[str], words: List[str]
    ) -> Tuple[int, List[str], Dict]:
        reasons = []
        score = 0
        alpha = [w for w in words if w.isalpha()]

        # Bigram repetition
        bigrams = list(zip(alpha, alpha[1:]))
        bg_ratio = self._ngram_repeat_ratio(bigrams)

        # Trigram repetition
        trigrams = list(zip(alpha, alpha[1:], alpha[2:]))
        tg_ratio = self._ngram_repeat_ratio(trigrams)

        # Sentence-opening repetition (first 3 words)
        openings = []
        for s in sentences:
            ws = s.lower().split()
            if len(ws) >= 3:
                openings.append(" ".join(ws[:3]))
        open_ratio = self._sequence_repeat_ratio(openings)

        # AI hallmark phrase count
        hallmark_hits = []
        for sent in sentences:
            for pat in self._AI_RE:
                if pat.search(sent):
                    hallmark_hits.append(sent.strip())
                    break
        hallmark_ratio = len(hallmark_hits) / max(len(sentences), 1)

        if bg_ratio > 0.15:
            score += 4
            reasons.append(f"✔ High bigram repetition ({bg_ratio:.2%}) — repeated word pairs typical of AI")
        if tg_ratio > 0.08:
            score += 4
            reasons.append(f"✔ High trigram repetition ({tg_ratio:.2%}) — repeated 3-word patterns")
        if open_ratio > 0.20:
            score += 4
            reasons.append(f"✔ Repeated sentence starters ({open_ratio:.2%}) — formulaic opening structure")
        elif open_ratio > 0.10:
            score += 2

        if hallmark_ratio > 0.25:
            score += 3
            reasons.append(
                f"✔ Frequent AI hallmark phrases ({len(hallmark_hits)} sentences) — "
                f"e.g. 'furthermore', 'in conclusion', 'it is important to note'"
            )
        elif hallmark_ratio > 0.10:
            score += 1
            reasons.append(f"✔ Some AI hallmark phrases detected ({len(hallmark_hits)} sentences)")

        score = min(score, 15)
        feats = {
            "bigram_repeat_ratio": round(bg_ratio, 4),
            "trigram_repeat_ratio": round(tg_ratio, 4),
            "sentence_opener_repeat_ratio": round(open_ratio, 4),
            "hallmark_phrase_ratio": round(hallmark_ratio, 4),
            "hallmark_sentences": hallmark_hits[:5],
        }
        return score, reasons, feats

    @staticmethod
    def _ngram_repeat_ratio(ngrams: list) -> float:
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        repeated = sum(v for v in counts.values() if v > 1)
        return repeated / len(ngrams)

    @staticmethod
    def _sequence_repeat_ratio(items: list) -> float:
        if not items:
            return 0.0
        counts = Counter(items)
        repeated = sum(v for v in counts.values() if v > 1)
        return repeated / len(items)

    # ─────────────────────────────────────────────────────────
    # 5. Formal tone detection (max 15 pts)
    # ─────────────────────────────────────────────────────────

    def _formal_tone_heuristic(
        self, text: str, sentences: List[str], words: List[str]
    ) -> Tuple[int, List[str], Dict]:
        reasons = []
        score = 0
        total_sents = max(len(sentences), 1)
        alpha = [w for w in words if w.isalpha()]
        total_words = max(len(alpha), 1)

        # Passive voice density
        passive_count = len(self._PASSIVE.findall(text))
        passive_ratio = passive_count / total_sents
        if passive_ratio > 0.5:
            score += 5
            reasons.append(f"✔ High passive voice usage ({passive_ratio:.2f} per sentence) — formally structured text")
        elif passive_ratio > 0.25:
            score += 2

        # Transition word density
        text_lower = text.lower()
        transition_hits = sum(
            1 for tw in self._TRANSITION_WORDS if tw in text_lower
        )
        transition_ratio = transition_hits / total_sents
        if transition_ratio > 0.4:
            score += 5
            reasons.append(f"✔ Heavy use of transition words ({transition_hits} unique, ratio {transition_ratio:.2f}/sentence)")
        elif transition_ratio > 0.2:
            score += 2
            reasons.append(f"✔ Moderate transition word density ({transition_hits} unique)")

        # Academic vocabulary density
        academic_hits = sum(
            1 for w in alpha if w in self._ACADEMIC_VOCAB
        )
        academic_ratio = academic_hits / total_words
        if academic_ratio > 0.04:
            score += 5
            reasons.append(f"✔ High academic vocabulary density ({academic_ratio:.2%}) — overly formal register")
        elif academic_ratio > 0.02:
            score += 2
            reasons.append(f"✔ Moderate academic vocabulary ({academic_ratio:.2%})")

        score = min(score, 15)
        feats = {
            "passive_voice_ratio": round(passive_ratio, 4),
            "transition_word_count": transition_hits,
            "transition_ratio_per_sentence": round(transition_ratio, 4),
            "academic_vocab_ratio": round(academic_ratio, 4),
        }
        return score, reasons, feats

    # ─────────────────────────────────────────────────────────
    # 6. Human imperfection signals (max 10 pts deduction)
    # ─────────────────────────────────────────────────────────

    def _human_signal_heuristic(
        self, text: str, words: List[str]
    ) -> Tuple[int, List[str], Dict]:
        """
        Detects signals that indicate human writing.
        Returns a *deduction* from the AI score (0-10).
        """
        reasons = []
        deduction = 0

        contractions = len(self._CONTRACTIONS.findall(text))
        hedges = len(self._HEDGE_WORDS.findall(text))
        emotional = len(self._EMOTIONAL_MARKERS.findall(text))
        informal = len(self._INFORMAL_MARKERS.findall(text))

        total_signals = contractions + hedges + emotional + informal

        if contractions > 2:
            deduction += 3
            reasons.append(f"✘ {contractions} contractions found — informal register typical of human writing")
        elif contractions > 0:
            deduction += 1

        if hedges > 1:
            deduction += 3
            reasons.append(f"✘ {hedges} hedging/opinion markers ('I think', 'maybe', 'honestly') — personal voice present")
        elif hedges > 0:
            deduction += 1

        if emotional > 1:
            deduction += 2
            reasons.append(f"✘ {emotional} emotional markers — expressive language typical of human writing")

        if informal > 1:
            deduction += 2
            reasons.append(f"✘ {informal} informal markers — casual language inconsistent with AI style")

        if total_signals == 0:
            # Complete absence of human signals is itself a signal
            reasons.append("✔ No contractions, hedges, personal opinions, or emotional language detected")

        deduction = min(deduction, 10)
        feats = {
            "contractions_count": contractions,
            "hedge_words_count": hedges,
            "emotional_markers_count": emotional,
            "informal_markers_count": informal,
            "human_signal_total": total_signals,
        }
        return deduction, reasons, feats


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

_HEURISTIC_ENGINE = HeuristicEngine()


# ─────────────────────────────────────────────────────────────
# Signal 2: Transformer classifier scoring (chunked)
# ─────────────────────────────────────────────────────────────

def transformer_score(text: str) -> Dict:
    if not HAS_CLASSIFIER:
        return {"probability": 0, "available": False}

    chunks = chunk_text(text)
    scores = []

    for chunk in chunks:
        try:
            result = _CLASSIFIER(chunk, truncation=True, max_length=512)[0]
            label = result["label"].lower()
            score = float(result["score"])
            scores.append(score * 100 if ("ai" in label or "fake" in label) else (1 - score) * 100)
        except Exception:
            continue

    if not scores:
        return {"probability": 0, "available": False}

    return {
        "probability": int(np.mean(scores)),
        "available": True,
    }


# ─────────────────────────────────────────────────────────────
# Signal 3: GPT-2 perplexity
# ─────────────────────────────────────────────────────────────

def calculate_perplexity(text: str) -> float:
    """
    Lower perplexity = text is more predictable = likely AI-generated.
    Human text: typically perplexity > 50.
    AI text:    typically perplexity < 30.
    Returns 999.0 if GPT-2 is unavailable.
    """
    if not HAS_PERPLEXITY:
        return 999.0

    try:
        import torch
        encodings = _PPL_TOKENIZER(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = encodings.input_ids

        with torch.no_grad():
            outputs = _PPL_MODEL(input_ids, labels=input_ids)

        return float(torch.exp(outputs.loss).item())

    except Exception:
        return 999.0


def perplexity_ai_score(perplexity: float) -> int:
    """Convert perplexity to an AI likelihood score (0-35)."""
    if perplexity == 999.0:
        return 0
    if perplexity < 20:
        return 35
    if perplexity < 30:
        return 25
    if perplexity < 40:
        return 15
    return 0


# ─────────────────────────────────────────────────────────────
# Signal 4: Stylometric analysis (burstiness + diversity)
# ─────────────────────────────────────────────────────────────

def lexical_diversity(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def sentence_variance(sentences: List[str]) -> float:
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 0.0
    return float(np.var(lengths))


def burstiness(sentences: List[str]) -> float:
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 0.0
    mean = np.mean(lengths)
    std = np.std(lengths)
    return float(std / mean) if mean > 0 else 0.0


def stylometric_score(text: str, sentences: List[str]) -> Dict:
    diversity = lexical_diversity(text)
    variance = sentence_variance(sentences)
    burst = burstiness(sentences)

    score = 0

    if variance < 80:
        score += 20
    if variance < 40:
        score += 10

    if burst < 0.55:
        score += 20
    if burst < 0.40:
        score += 10

    if diversity < 0.60:
        score += 15
    if diversity < 0.45:
        score += 10

    return {
        "style_score": score,
        "lexical_diversity": round(diversity, 4),
        "sentence_variance": round(variance, 4),
        "burstiness": round(burst, 4),
    }


# ─────────────────────────────────────────────────────────────
# Signal 5: Semantic uniformity via SBERT
# ─────────────────────────────────────────────────────────────

def semantic_uniformity(sentences: List[str]) -> float:
    if len(sentences) < 3:
        return 0.0

    try:
        from embeddings.sbert_engine import EmbeddingEngine
        from sklearn.metrics.pairwise import cosine_similarity

        engine = EmbeddingEngine()
        embeddings = engine.encode_documents(sentences, batch_size=16)

        similarities = [
            float(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0])
            for i in range(len(embeddings) - 1)
        ]
        return float(np.mean(similarities))

    except Exception:
        return 0.0


def semantic_uniformity_score(uniformity: float) -> int:
    if uniformity > 0.92:
        return 20
    if uniformity > 0.88:
        return 12
    if uniformity > 0.82:
        return 6
    return 0


# ─────────────────────────────────────────────────────────────
# Signal 6: Repetition pattern detection (legacy simple scorer)
# ─────────────────────────────────────────────────────────────

def repetition_score(sentences: List[str]) -> float:
    openings = []
    for sent in sentences:
        words = sent.lower().split()
        if len(words) >= 3:
            openings.append(" ".join(words[:3]))

    if not openings:
        return 0.0

    counts = Counter(openings)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / len(openings)


# ─────────────────────────────────────────────────────────────
# Main analysis — weighted fusion of all signals
# ─────────────────────────────────────────────────────────────

def analyse_text_for_ai(text: str) -> Dict:
    """
    Full multi-signal AI detection pipeline with advanced heuristics.

    Returns:
        ai_probability  : 0-100 calibrated score
        verdict         : 'ai' | 'human' | 'uncertain'
        explanation     : list[str] — human-readable reasons from heuristic engine
        signals         : breakdown of every sub-signal
    """
    sentences = split_sentences(text)

    # ── Run all signals ────────────────────────────────────────
    transformer = transformer_score(text)
    style       = stylometric_score(text, sentences)
    perplexity  = calculate_perplexity(text)
    uniformity  = semantic_uniformity(sentences)
    repeat      = repetition_score(sentences)

    # Advanced heuristic engine (passes already-computed perplexity)
    heuristic   = _HEURISTIC_ENGINE.analyse(text, perplexity=perplexity)

    # ── Convert signals to scores ──────────────────────────────
    ppl_score   = perplexity_ai_score(perplexity)
    sem_score   = semantic_uniformity_score(uniformity)
    rep_score   = 10 if repeat > 0.20 else (5 if repeat > 0.10 else 0)

    # ── Weighted fusion ────────────────────────────────────────
    # Weights: transformer 50 %, heuristic 20 %, stylometric 15 %,
    #          perplexity 10 %, semantic+repeat 5 %
    final_probability = (
        transformer["probability"] * 0.50
        + heuristic["probability"]  * 0.20
        + style["style_score"]      * 0.15
        + ppl_score                 * 0.10
        + (sem_score + rep_score)   * 0.05
    )

    final_probability = int(min(100, max(0, final_probability)))

    # ── Verdict thresholds ─────────────────────────────────────
    if final_probability >= 75:
        verdict = "ai"
        verdict_label = "Highly AI Generated"
    elif final_probability >= 50:
        verdict = "ai"
        verdict_label = "Possibly AI Generated"
    elif final_probability >= 31:
        verdict = "uncertain"
        verdict_label = "Uncertain"
    else:
        verdict = "human"
        verdict_label = "Likely Human Written"

    return {
        "ai_probability": final_probability,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "explanation": heuristic["reasons"],   # ← explainable bullets
        "signals": {
            "transformer":              transformer,
            "heuristic": {
                "probability":          heuristic["probability"],
                "sub_scores":           heuristic["sub_scores"],
                "features":             heuristic["features"],
                "reasons":              heuristic["reasons"],
            },
            "stylometric":              style,
            "perplexity":               round(perplexity, 2) if perplexity != 999.0 else None,
            "perplexity_score":         ppl_score,
            "semantic_uniformity":      round(uniformity, 4),
            "semantic_score":           sem_score,
            "repetition_ratio":         round(repeat, 4),
            "repetition_score":         rep_score,
        },
        # Backward-compatibility keys
        "heuristic":  {"probability": heuristic["probability"], "flagged_sentences": heuristic["features"].get("hallmark_sentences", [])},
        "classifier": transformer,
    }