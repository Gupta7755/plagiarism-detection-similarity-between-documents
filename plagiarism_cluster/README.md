# PlagiarismShield — Local AI Detection Suite

A full-stack plagiarism detection system using SBERT embeddings, FAISS retrieval,
DBSCAN clustering, and local AI-text detection. **Zero external API calls.**

---

## Supported Datasets

Place these zip files inside a `data/` folder next to `manage.py`.
The system auto-detects and extracts them automatically.

| Dataset | File(s) | What it provides |
|---|---|---|
| **PAN-2011** *(most important)* | `pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011.part1` + `.part2` | External + intrinsic detection, XML ground truth |
| **PAN-2025 AI Plagiarism** | `pan25-generated-plagiarism-detection-spot-checks/` (3 sub-zips) | AI vs human labels (JSONL) |
| **ExAra PAN-2015** | `ExAracorpusPAN2015.zip` | Arabic multilingual, XML annotations |
| **Project Gutenberg** | `The Project Gutenberg eBook of The*.zip` | Reference source corpus |

### Archive structures handled automatically

```
data/
├── ExAracorpusPAN2015.zip                       ← double-nested zip
├── pan25-generated-plagiarism-detection-spot-checks/
│   ├── pan25-...-spot-check.zip
│   ├── pan25-...-train.zip                      ← 23 GB, skipped by default
│   └── pan25-...-validation.zip
├── pan-plagiarism-corpus-2011/
│   ├── pan-plagiarism-corpus-2011.part1         ← multi-part archive
│   └── pan-plagiarism-corpus-2011.part2
└── The Project Gutenberg eBook of The*.zip
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
python scripts/setup_nltk.py
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set DATASET_DIR to your data folder
```

### 3. Set up database

```bash
python manage.py migrate
```

### 4. Run the server

```bash
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser.

---

## Project Structure

```
plagiarism_detector/
├── ingestion/           Dataset loaders for all 4 corpora (handles real archive structures)
│   └── data_loader.py   PAN2015Loader, PAN2011Loader, PAN25Loader, GutenbergLoader
├── preprocessing/       Text cleaning, sentence splitting, language detection
├── deduplication/       MinHash + LSH near-duplicate removal
├── embeddings/          SBERT all-MiniLM-L6-v2 (local), TF-IDF fallback
├── retrieval/           FAISS IndexFlatIP ANN search
├── clustering/          DBSCAN on cosine distance
├── detection/           Sentence-level span alignment + local AI detector
│   ├── span_detector.py Character-offset span detection
│   └── ai_detector.py   Heuristic + local HuggingFace classifier
├── evaluation/          PAN XML output + Precision/Recall/F1/Granularity
├── pipeline_core/       Main orchestrator (no API calls)
├── versioning/          Document hash + reproducibility tracking
├── api/                 Django REST endpoints (all local)
├── config/              Django settings
├── templates/           Single-page frontend
├── scripts/
│   ├── run_pipeline.py  CLI entry-point
│   └── setup_nltk.py    Download NLTK data
└── manage.py
```

---

## API Endpoints

| Method | URL | Description |
|---|---|---|
| `GET`  | `/api/health/`       | Health check |
| `POST` | `/api/similarity/`   | Compare two documents (multipart: `doc_a`, `doc_b`) |
| `POST` | `/api/upload/`       | Batch corpus upload (multipart: `documents[]`) |
| `POST` | `/api/ai-detect/`    | Local AI-text detection (JSON: `{"text":"..."}`) |
| `POST` | `/api/dataset-run/`  | Run pipeline on server DATASET_DIR |
| `POST` | `/api/evaluate/`     | Score against PAN ground truth |

---

## CLI Usage

```bash
# Run pipeline on datasets, write PAN XML output
python scripts/run_pipeline.py --data-dir ./data --output-dir ./pan_output

# Evaluate against PAN-2011 ground truth
python scripts/run_pipeline.py --data-dir ./data --evaluate pan2011

# Include PAN-2025 train split (23 GB — large!)
python scripts/run_pipeline.py --data-dir ./data --load-train
```

---

## How It Works (No External APIs)

```
Documents
   ↓
[Preprocessing]   NLTK sentence splitting, unicode normalization
   ↓
[Deduplication]   MinHash LSH (datasketch) — removes near-duplicates
   ↓
[Embedding]       SBERT all-MiniLM-L6-v2 — 384-dim local vectors
   ↓
[FAISS Index]     Cosine-similarity ANN search
   ↓
[DBSCAN]          Cluster similar documents (eps=0.2 ≈ sim>0.80)
   ↓
[Span Detection]  Sentence-pair cosine → character offsets
   ↓
[AI Detection]    Heuristic patterns + local roberta-base-openai-detector
   ↓
[PAN XML Output]  Standard evaluation format
   ↓
[Evaluator]       Precision / Recall / F1 / Granularity
```

---

## Notes

- The PAN-2025 **train** split (~23 GB uncompressed) is skipped by default to avoid
  exhausting RAM. Pass `load_train=True` to `DataLoader` or `--load-train` on the CLI.
- SBERT model (`all-MiniLM-L6-v2`) downloads once on first run (~90 MB) from HuggingFace.
  After that it is cached locally and works offline.
- The local AI detector uses `roberta-base-openai-detector` if you have it cached
  (`~500 MB`). It falls back automatically to the fast heuristic scorer.
- SQLite is used for the Django session/admin database — no PostgreSQL needed.
