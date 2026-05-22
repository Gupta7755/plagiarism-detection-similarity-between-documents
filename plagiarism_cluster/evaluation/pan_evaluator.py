"""
evaluation/pan_evaluator.py
============================
PAN XML output + full evaluation metrics (Precision, Recall, F1, Granularity).
Also includes AI-detection evaluator for PAN-2025.
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  XML Output
# ─────────────────────────────────────────────────────────────────────────────

class PANOutputGenerator:
    @staticmethod
    def generate_xml(doc_name: str, source_ref: str, spans: List[Dict], output_path: str):
        root = ET.Element("document", reference=doc_name)
        for span in spans:
            ET.SubElement(root, "feature",
                name="plagiarism",
                this_offset=str(span.get("this_offset", 0)),
                this_length=str(span.get("this_length", 0)),
                source_reference=source_ref,
                source_offset=str(span.get("source_offset", 0)),
                source_length=str(span.get("source_length", 0)),
            )
        pretty = xml.dom.minidom.parseString(
            ET.tostring(root, encoding="utf-8")
        ).toprettyxml(indent="  ")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty)


# ─────────────────────────────────────────────────────────────────────────────
#  Character-level Span Evaluation (PAN Protocol)
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Implements the official PAN plagiarism detection evaluation:
      Precision, Recall, F1, Granularity at character level.
    """

    @staticmethod
    def _chars(offset: int, length: int) -> set:
        return set(range(offset, offset + max(0, length)))

    @classmethod
    def evaluate(
        cls,
        predicted: List[Dict],
        ground_truth: List[Dict],
    ) -> Dict[str, float]:
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "granularity": 1.0}

        pred_sets = [cls._chars(s["this_offset"], s["this_length"]) for s in predicted]
        gt_sets   = [cls._chars(s["this_offset"], s["this_length"]) for s in ground_truth]

        pred_union = set().union(*pred_sets) if pred_sets else set()
        gt_union   = set().union(*gt_sets)

        tp        = len(pred_union & gt_union)
        precision = tp / len(pred_union) if pred_union else 0.0
        recall    = tp / len(gt_union)   if gt_union   else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        # Granularity: mean number of predicted spans covering each GT span
        gran_vals = []
        for gt_set in gt_sets:
            covering = sum(1 for ps in pred_sets if ps & gt_set)
            gran_vals.append(max(1, covering))
        granularity = sum(gran_vals) / len(gran_vals)

        return {
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "f1":          round(f1, 4),
            "granularity": round(granularity, 4),
        }

    @classmethod
    def evaluate_corpus(
        cls,
        predictions: Dict[str, List[Dict]],
        ground_truth: Dict[str, List[Dict]],
    ) -> Dict[str, float]:
        all_docs = set(ground_truth) | set(predictions)
        results  = [
            cls.evaluate(predictions.get(d, []), ground_truth.get(d, []))
            for d in all_docs
        ]
        if not results:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "granularity": 1.0}
        keys = results[0].keys()
        return {k: round(sum(r[k] for r in results) / len(results), 4) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
#  AI-Detection Evaluation (PAN-2025)
# ─────────────────────────────────────────────────────────────────────────────

class AIDetectionEvaluator:
    @staticmethod
    def evaluate(
        predicted: Dict[str, str],
        ground_truth: Dict[str, str],
    ) -> Dict:
        common = set(predicted) & set(ground_truth)
        if not common:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

        tp = fp = fn = tn = 0
        for doc_id in common:
            p = predicted[doc_id].lower()
            t = ground_truth[doc_id].lower()
            if   p == "ai"  and t == "ai":   tp += 1
            elif p == "ai"  and t != "ai":   fp += 1
            elif p != "ai"  and t == "ai":   fn += 1
            else:                             tn += 1

        n         = len(common)
        accuracy  = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        return {
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
