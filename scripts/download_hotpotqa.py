"""Download the HotpotQA dev (distractor) JSON to data/raw/hotpotqa/.

The HuggingFace datasets mirror of HotpotQA broke in datasets>=4.x because
the dataset uses the old single-name format. We pin to the canonical JSON
from the HotpotQA project page for reproducibility.

Run from repo root:
    python scripts/download_hotpotqa.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

# Canonical dev-distractor split from the HotpotQA project page.
# (Yang et al. 2018; project page: https://hotpotqa.github.io/)
_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
_LOCAL = Path("data/raw/hotpotqa/hotpot_dev_distractor_v1.json")


def main() -> None:
    _LOCAL.parent.mkdir(parents=True, exist_ok=True)

    if _LOCAL.exists():
        size_mb = _LOCAL.stat().st_size / (1024 * 1024)
        print(f"Already downloaded: {_LOCAL} ({size_mb:.1f} MB)")
        return

    print(f"Downloading {_URL} ...")
    with requests.get(_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        h = hashlib.sha256()
        with _LOCAL.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
                    h.update(chunk)
    size_mb = _LOCAL.stat().st_size / (1024 * 1024)
    print(f"Saved to {_LOCAL} ({size_mb:.1f} MB)")
    print(f"SHA-256: {h.hexdigest()}")


if __name__ == "__main__":
    main()
