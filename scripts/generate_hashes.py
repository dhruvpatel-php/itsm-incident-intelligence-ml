import json
import hashlib
from pathlib import Path

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = Path(__file__).resolve().parent.parent
    hashes = {
        "metadata": {
            "description": "Cryptographic SHA-256 hashes of models, datasets, reports and notebooks"
        },
        "models": {},
        "reports": {},
        "data": {},
        "notebooks": {}
    }

    models_dir = root / "models"
    if models_dir.exists():
        for p in sorted(models_dir.glob("*")):
            if p.is_file():
                hashes["models"][p.name] = {
                    "sha256": get_file_sha256(p),
                    "size_bytes": p.stat().st_size
                }

    reports_dir = root / "reports"
    if reports_dir.exists():
        for p in sorted(reports_dir.glob("*.json")):
            if p.name != "artifact_hashes.json":
                hashes["reports"][p.name] = {
                    "sha256": get_file_sha256(p),
                    "size_bytes": p.stat().st_size
                }

    raw_data = root / "dataset_list.csv"
    if raw_data.exists():
        hashes["data"]["dataset_list.csv"] = {
            "sha256": get_file_sha256(raw_data),
            "size_bytes": raw_data.stat().st_size
        }

    for p in sorted(root.glob("*.ipynb")):
        hashes["notebooks"][p.name] = {
            "sha256": get_file_sha256(p),
            "size_bytes": p.stat().st_size
        }

    out_file = reports_dir / "artifact_hashes.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print(f"Generated {out_file} with {len(hashes['models'])} models and {len(hashes['reports'])} reports.")

if __name__ == "__main__":
    main()
