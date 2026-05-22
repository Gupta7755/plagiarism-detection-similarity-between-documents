"""
versioning/tracker.py
=====================
Tracks document hashes and model versions for reproducibility.
"""
import hashlib, json, os
from datetime import datetime

class VersionTracker:
    def __init__(self, log_path: str = "version_log.json"):
        self.log_path = log_path
        self._log = self._load()

    def _load(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"model": {}, "docs": {}}

    def _save(self):
        try:
            with open(self.log_path, "w") as f:
                json.dump(self._log, f, indent=2)
        except Exception:
            pass

    def register_doc(self, doc_id: str, text: str):
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        self._log["docs"][doc_id] = {"hash": h, "ts": datetime.utcnow().isoformat()}
        self._save()
        return h

    def register_model(self, name: str, version: str):
        self._log["model"][name] = {"version": version, "ts": datetime.utcnow().isoformat()}
        self._save()

    def get_doc_hash(self, doc_id: str) -> str:
        return self._log.get("docs", {}).get(doc_id, {}).get("hash", "")
