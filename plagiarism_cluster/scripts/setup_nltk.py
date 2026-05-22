#!/usr/bin/env python3
"""Download required NLTK data packages."""
import nltk
for pkg in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"):
    print(f"Downloading {pkg}...")
    nltk.download(pkg, quiet=False)
print("NLTK setup complete.")
