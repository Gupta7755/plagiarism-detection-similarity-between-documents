"""
ingestion/data_loader.py
========================
Dataset-specific loaders for all 4 corpora, matching the EXACT
archive structures visible in the screenshots:

  1. ExAracorpusPAN2015
       ExAracorpusPAN2015.zip
         └── ExAraCorpusPAN2015/     (inner folder)
               ├── suspicious-document/  *.txt + *.xml
               └── source-document/      *.txt

  2. pan-plagiarism-corpus-2011
       pan-plagiarism-corpus-2011/
         ├── pan-plagiarism-corpus-2011.part1  (multi-part zip)
         └── pan-plagiarism-corpus-2011.part2
       After join+extract:
         external-detection-corpus/
           ├── suspicious-document/  *.txt + *.xml
           └── source-document/      *.txt
         intrinsic-detection-corpus/
           └── suspicious-document/  *.txt + *.xml

  3. pan25-generated-plagiarism-detection-spot-checks
       Three sub-zips inside the outer zip:
         pan25-generated-plagiarism-detection-spot-check.zip
         pan25-generated-plagiarism-detection-train.zip      (large)
         pan25-generated-plagiarism-detection-validation.zip
       Each contains JSONL files with {"id":…,"text":…,"label":"human"|"ai"}

  4. The Project Gutenberg eBook of The*.zip
       Single plain-text .txt file — boilerplate stripped automatically
"""

import os
import re
import glob
import json
import zipfile
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_txt(path: str) -> str:
    for enc in ("utf-8", "latin-1", "cp1252", "utf-16"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def _safe_extract(zip_path: str, dest: str) -> bool:
    """Extract zip safely; return True on success."""
    try:
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        return True
    except (zipfile.BadZipFile, struct.error, Exception) as e:
        print(f"  [!] Cannot extract {zip_path}: {e}")
        return False


def _walk_txts(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))


def _walk_xmls(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "**", "*.xml"), recursive=True))


def _parse_pan_xml(xml_path: str) -> List[Dict]:
    """Parse PAN ground-truth XML and return list of annotation dicts."""
    annotations = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        doc_ref = root.get("reference", Path(xml_path).stem)
        for feat in root.findall(".//feature[@name='plagiarism']"):
            annotations.append({
                "suspicious_doc":  doc_ref,
                "source_doc":      feat.get("source_reference", ""),
                "this_offset":     int(feat.get("this_offset",  0)),
                "this_length":     int(feat.get("this_length",  0)),
                "source_offset":   int(feat.get("source_offset", 0)),
                "source_length":   int(feat.get("source_length", 0)),
                "obfuscation":     feat.get("obfuscation", feat.get("type", "none")),
            })
    except ET.ParseError:
        pass
    return annotations


# ─────────────────────────────────────────────────────────────────────────────
#  1. ExAraCorpusPAN2015
# ─────────────────────────────────────────────────────────────────────────────

class PAN2015Loader:
    """
    Handles the double-nested structure:
      ExAracorpusPAN2015.zip → ExAraCorpusPAN2015/ → suspicious-document/ + source-document/
    """

    def __init__(self, dataset_path: str, extract_base: str):
        """
        dataset_path : path to ExAracorpusPAN2015.zip  OR  already-extracted folder
        extract_base : where to put extracted files
        """
        self.extract_base = extract_base
        self.root = self._resolve_root(dataset_path)

    def _resolve_root(self, path: str) -> str:
        # If it's a zip, extract it first
        if path.endswith(".zip") and os.path.isfile(path):
            dest = os.path.join(self.extract_base, "ExAracorpusPAN2015")
            if not os.path.isdir(dest):
                print(f"  [PAN2015] Extracting {path} ...")
                _safe_extract(path, dest)
            path = dest

        # Navigate into the inner ExAraCorpusPAN2015 folder if present
        for name in os.listdir(path) if os.path.isdir(path) else []:
            candidate = os.path.join(path, name)
            if os.path.isdir(candidate) and "ExAra" in name:
                # Check if this itself contains the document folders
                sub_dirs = os.listdir(candidate) if os.path.isdir(candidate) else []
                if "suspicious-document" in sub_dirs or "source-document" in sub_dirs:
                    return candidate
                # Maybe one level deeper (double zip)
                for sub in sub_dirs:
                    deeper = os.path.join(candidate, sub)
                    if os.path.isdir(deeper):
                        dd = os.listdir(deeper)
                        if "suspicious-document" in dd or "source-document" in dd:
                            return deeper
                return candidate
        return path

    def load_suspicious(self) -> Dict[str, str]:
        docs: Dict[str, str] = {}
        folder = os.path.join(self.root, "suspicious-document")
        if not os.path.isdir(folder):
            folder = self.root
        for p in _walk_txts(folder):
            docs["pan15_susp_" + Path(p).stem] = _read_txt(p)
        print(f"  [PAN2015] {len(docs)} suspicious docs loaded.")
        return docs

    def load_sources(self) -> Dict[str, str]:
        docs: Dict[str, str] = {}
        folder = os.path.join(self.root, "source-document")
        if not os.path.isdir(folder):
            return docs
        for p in _walk_txts(folder):
            docs["pan15_src_" + Path(p).stem] = _read_txt(p)
        print(f"  [PAN2015] {len(docs)} source docs loaded.")
        return docs

    def load_ground_truth(self) -> List[Dict]:
        annotations = []
        folder = os.path.join(self.root, "suspicious-document")
        if not os.path.isdir(folder):
            folder = self.root
        for p in _walk_xmls(folder):
            annotations.extend(_parse_pan_xml(p))
        print(f"  [PAN2015] {len(annotations)} ground-truth annotations.")
        return annotations

    def load_all(self) -> Dict[str, str]:
        docs = {}
        docs.update(self.load_suspicious())
        docs.update(self.load_sources())
        return docs


# ─────────────────────────────────────────────────────────────────────────────
#  2. PAN Plagiarism Corpus 2011
# ─────────────────────────────────────────────────────────────────────────────

class PAN2011Loader:
    """
    Handles the multi-part archive:
      pan-plagiarism-corpus-2011/
        pan-plagiarism-corpus-2011.part1   (RAR/zip part)
        pan-plagiarism-corpus-2011.part2
    After extraction:
      external-detection-corpus/
        suspicious-document/  *.txt + *.xml
        source-document/      *.txt
      intrinsic-detection-corpus/
        suspicious-document/  *.txt + *.xml
    """

    def __init__(self, dataset_path: str, extract_base: str):
        self.extract_base = extract_base
        self.root = self._resolve_root(dataset_path)

    def _resolve_root(self, path: str) -> str:
        if not os.path.isdir(path):
            return path

        # Check if already extracted (contains the corpus subdirs)
        for dirpath, dirnames, _ in os.walk(path):
            if "external-detection-corpus" in dirnames or "intrinsic-detection-corpus" in dirnames:
                return dirpath

        # Try to extract part1 as zip (sometimes .part1 is a zip)
        dest = os.path.join(self.extract_base, "pan2011_extracted")
        if not os.path.isdir(dest):
            for f in sorted(glob.glob(os.path.join(path, "**", "*.part1"), recursive=True)):
                print(f"  [PAN2011] Attempting extraction of {f} ...")
                ok = _safe_extract(f, dest)
                if ok:
                    break
            # Also try any .zip in the folder
            if not os.path.isdir(dest) or not os.listdir(dest):
                for f in sorted(glob.glob(os.path.join(path, "**", "*.zip"), recursive=True)):
                    print(f"  [PAN2011] Extracting {f} ...")
                    _safe_extract(f, dest)
        return dest if os.path.isdir(dest) else path

    def _find_subdir(self, name: str) -> Optional[str]:
        for dirpath, dirnames, _ in os.walk(self.root):
            if name in dirnames:
                return os.path.join(dirpath, name)
        return None

    def _load_txts(self, folder: str, prefix: str) -> Dict[str, str]:
        docs: Dict[str, str] = {}
        if folder and os.path.isdir(folder):
            for p in _walk_txts(folder):
                docs[prefix + Path(p).stem] = _read_txt(p)
        return docs

    def load_external(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        ext = self._find_subdir("external-detection-corpus")
        if not ext:
            return {}, {}
        susp = self._load_txts(os.path.join(ext, "suspicious-document"), "pan11_ext_susp_")
        src  = self._load_txts(os.path.join(ext, "source-document"),     "pan11_ext_src_")
        print(f"  [PAN2011] External: {len(susp)} suspicious, {len(src)} source docs.")
        return susp, src

    def load_intrinsic(self) -> Dict[str, str]:
        intr = self._find_subdir("intrinsic-detection-corpus")
        if not intr:
            return {}
        docs = self._load_txts(os.path.join(intr, "suspicious-document"), "pan11_intr_susp_")
        print(f"  [PAN2011] Intrinsic: {len(docs)} suspicious docs.")
        return docs

    def load_ground_truth(self, task: str = "external") -> List[Dict]:
        annotations = []
        if task == "external":
            base = self._find_subdir("external-detection-corpus")
            folder = os.path.join(base, "suspicious-document") if base else None
        else:
            base = self._find_subdir("intrinsic-detection-corpus")
            folder = os.path.join(base, "suspicious-document") if base else None
        if folder and os.path.isdir(folder):
            for p in _walk_xmls(folder):
                annotations.extend(_parse_pan_xml(p))
        print(f"  [PAN2011] {len(annotations)} GT annotations ({task}).")
        return annotations

    def load_all(self) -> Dict[str, str]:
        docs = {}
        susp, src = self.load_external()
        docs.update(susp)
        docs.update(src)
        docs.update(self.load_intrinsic())
        return docs


# ─────────────────────────────────────────────────────────────────────────────
#  3. PAN-2025 Generated Plagiarism
# ─────────────────────────────────────────────────────────────────────────────

class PAN25Loader:
    """
    Handles three sub-zips found inside the outer zip:
      pan25-generated-plagiarism-detection-spot-check.zip
      pan25-generated-plagiarism-detection-train.zip      (very large)
      pan25-generated-plagiarism-detection-validation.zip

    Each inner zip contains JSONL files:
      {"id": "...", "text": "...", "label": "human" | "ai",
       "model": "llama-3-...", ...}

    The TRAIN split is 23 GB uncompressed — we only use spot-check
    and validation by default. Set load_train=True to include train.
    """

    SUB_SPLITS = {
        "spot-check":  "spot-check",
        "validation":  "validation",
        "train":       "train",   # large — off by default
    }

    def __init__(self, dataset_path: str, extract_base: str, load_train: bool = False):
        self.extract_base = extract_base
        self.load_train   = load_train
        self.root = dataset_path   # outer zip or already-extracted folder

    def _get_split_roots(self) -> Dict[str, str]:
        """Return { split_name: extracted_folder } for each sub-zip found."""
        split_roots: Dict[str, str] = {}

        # First check if already extracted
        if os.path.isdir(self.root):
            for name in os.listdir(self.root):
                p = os.path.join(self.root, name)
                if os.path.isdir(p):
                    for key in self.SUB_SPLITS:
                        if key in name.lower():
                            split_roots[key] = p

            # Also search for inner zips to extract
            for zip_path in glob.glob(os.path.join(self.root, "**", "*.zip"), recursive=True):
                zip_name = Path(zip_path).stem.lower()
                for key in self.SUB_SPLITS:
                    if key in zip_name:
                        dest = os.path.join(self.extract_base, f"pan25_{key}")
                        if not os.path.isdir(dest) or not os.listdir(dest):
                            if key == "train" and not self.load_train:
                                print(f"  [PAN25] Skipping large train split (set load_train=True to include).")
                                continue
                            print(f"  [PAN25] Extracting {zip_path} ...")
                            _safe_extract(zip_path, dest)
                        split_roots[key] = dest
        return split_roots

    def load(self, max_per_split: int = 5000) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Returns (docs, labels) where labels[doc_id] = 'human' | 'ai' | 'unknown'
        max_per_split limits how many docs per split to load (prevents RAM exhaustion)
        """
        docs:   Dict[str, str] = {}
        labels: Dict[str, str] = {}

        split_roots = self._get_split_roots()
        if not split_roots:
            print("  [PAN25] No split folders found. Check dataset path.")
            return docs, labels

        for split_name, split_root in split_roots.items():
            if split_name == "train" and not self.load_train:
                continue

            count = 0
            for jsonl_path in sorted(glob.glob(os.path.join(split_root, "**", "*.jsonl"), recursive=True)):
                with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text  = obj.get("text",  obj.get("content", ""))
                        label = obj.get("label", obj.get("generated", "unknown"))
                        if isinstance(label, bool):
                            label = "ai" if label else "human"
                        label = str(label).lower()
                        if text and len(text.split()) >= 10:
                            doc_id = f"pan25_{split_name}_{obj.get('id', count)}"
                            docs[doc_id]   = text
                            labels[doc_id] = label
                            count += 1
                            if count >= max_per_split:
                                break
                if count >= max_per_split:
                    break

            ai_n    = sum(1 for v in labels.values() if v == "ai"    and split_name in "")
            human_n = sum(1 for v in labels.values() if v == "human" and split_name in "")
            print(f"  [PAN25] Split '{split_name}': {count} docs loaded.")

        ai_total    = sum(1 for v in labels.values() if v == "ai")
        human_total = sum(1 for v in labels.values() if v == "human")
        print(f"  [PAN25] Total: {len(docs)} docs — {ai_total} AI, {human_total} human.")
        return docs, labels

    def load_all(self) -> Dict[str, str]:
        docs, _ = self.load()
        return docs


# ─────────────────────────────────────────────────────────────────────────────
#  4. Project Gutenberg
# ─────────────────────────────────────────────────────────────────────────────

class GutenbergLoader:
    """
    Strips Gutenberg header/footer boilerplate and chunks long texts
    into overlapping 3000-char windows for FAISS indexing.
    """

    _HDR = re.compile(
        r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*{3}",
        re.IGNORECASE
    )
    _FTR = re.compile(
        r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK",
        re.IGNORECASE
    )

    def __init__(self, dataset_path: str, extract_base: str):
        self.extract_base = extract_base
        self.root = self._resolve_root(dataset_path)

    def _resolve_root(self, path: str) -> str:
        if path.endswith(".zip") and os.path.isfile(path):
            dest = os.path.join(self.extract_base, "gutenberg")
            if not os.path.isdir(dest):
                print(f"  [Gutenberg] Extracting {path} ...")
                _safe_extract(path, dest)
            return dest
        return path

    @classmethod
    def strip(cls, text: str) -> str:
        m = cls._HDR.search(text)
        if m:
            text = text[m.end():]
        m = cls._FTR.search(text)
        if m:
            text = text[: m.start()]
        return text.strip()

    @classmethod
    def chunk(cls, text: str, size: int = 3000, overlap: int = 500) -> List[str]:
        step   = size - overlap
        chunks = []
        start  = 0
        while start < len(text):
            chunks.append(text[start: start + size])
            start += step
        return chunks

    def load(self, chunk: bool = True) -> Dict[str, str]:
        docs: Dict[str, str] = {}
        txt_files = _walk_txts(self.root)
        if not txt_files and os.path.isfile(self.root):
            txt_files = [self.root]
        for path in txt_files:
            raw  = _read_txt(path)
            body = self.strip(raw)
            stem = Path(path).stem
            if chunk:
                for i, piece in enumerate(self.chunk(body)):
                    docs[f"gutenberg_{stem}_c{i:04d}"] = piece
            else:
                docs[f"gutenberg_{stem}"] = body
        print(f"  [Gutenberg] {len(txt_files)} file(s) -> {len(docs)} chunks.")
        return docs

    def load_all(self) -> Dict[str, str]:
        return self.load(chunk=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Master DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Auto-detects the four datasets in data_dir and routes each to its loader.

    Expected layout:
        data/
          ExAracorpusPAN2015.zip      (or extracted folder)
          pan25-generated-plagiarism-detection-spot-checks/
            pan25-...-spot-check.zip
            pan25-...-train.zip
            pan25-...-validation.zip
          pan-plagiarism-corpus-2011/
            pan-plagiarism-corpus-2011.part1
            pan-plagiarism-corpus-2011.part2
          The Project Gutenberg eBook of The*.zip
    """

    def __init__(self, data_dir: str, load_train: bool = False):
        self.data_dir    = data_dir
        self.load_train  = load_train
        self._work       = os.path.join(data_dir, "_work")
        os.makedirs(self._work, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def load_all(self) -> Dict[str, str]:
        corpus: Dict[str, str] = {}
        for key, loader in self._discover().items():
            try:
                docs = loader.load_all()
                print(f"[DataLoader] {key}: {len(docs)} docs")
                corpus.update(docs)
            except Exception as e:
                print(f"[DataLoader] ERROR in {key}: {e}")
        if not corpus:
            corpus = self._generic_txt_load()
        return corpus

    def load_pan25_with_labels(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        p = self._find("pan25")
        if p:
            return PAN25Loader(p, self._work, self.load_train).load()
        return {}, {}

    def load_ground_truth(self, dataset: str = "pan2011", task: str = "external") -> List[Dict]:
        if dataset == "pan2011":
            p = self._find("pan2011")
            if p:
                return PAN2011Loader(p, self._work).load_ground_truth(task)
        elif dataset == "pan2015":
            p = self._find("pan2015")
            if p:
                return PAN2015Loader(p, self._work).load_ground_truth()
        return []

    # ── discovery ─────────────────────────────────────────────────────────────

    def _discover(self) -> Dict[str, object]:
        loaders = {}
        p15  = self._find("pan2015")
        p11  = self._find("pan2011")
        p25  = self._find("pan25")
        pgut = self._find("gutenberg")
        if p15:  loaders["pan2015"]   = PAN2015Loader(p15, self._work)
        if p11:  loaders["pan2011"]   = PAN2011Loader(p11, self._work)
        if p25:  loaders["pan25"]     = PAN25Loader(p25, self._work, self.load_train)
        if pgut: loaders["gutenberg"] = GutenbergLoader(pgut, self._work)
        return loaders

    def _find(self, key: str) -> Optional[str]:
        patterns = {
            "pan2015":   ["ExAracorpusPAN2015", "ExAra", "pan15"],
            "pan2011":   ["pan-plagiarism-corpus-2011", "pan2011", "pan11"],
            "pan25":     ["pan25-generated", "pan25", "pan-2025"],
            "gutenberg": ["gutenberg", "Gutenberg", "Project_Gutenberg"],
        }
        sigs = patterns.get(key, [])

        for root_dir, dirs, files in os.walk(self.data_dir):
            # Check directory names
            bname = os.path.basename(root_dir).lower()
            for sig in sigs:
                if sig.lower() in bname:
                    return root_dir
            # Check zip file names
            for f in files:
                for sig in sigs:
                    if sig.lower() in f.lower() and f.endswith(".zip"):
                        return os.path.join(root_dir, f)
        return None

    def _generic_txt_load(self) -> Dict[str, str]:
        docs: Dict[str, str] = {}
        for p in _walk_txts(self.data_dir):
            docs["doc_" + Path(p).stem] = _read_txt(p)
        print(f"[DataLoader] Generic fallback: {len(docs)} txt files.")
        return docs
