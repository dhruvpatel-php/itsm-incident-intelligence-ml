import re

from src.config import PROJECT_ROOT

DOC_FILES = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "REPORT.md",
    PROJECT_ROOT / "models" / "MODEL_CARD.md"
]

def test_no_file_urls():
    for doc in DOC_FILES:
        if not doc.exists():
            continue
        with open(doc, "r", encoding="utf-8") as f:
            content = f.read()
        file_urls = re.findall(r"file:///", content, re.IGNORECASE)
        assert len(file_urls) == 0, f"Found {len(file_urls)} 'file:///' URLs in {doc.name}"

def test_no_private_ips():
    for doc in DOC_FILES:
        if not doc.exists():
            continue
        with open(doc, "r", encoding="utf-8") as f:
            content = f.read()
        assert "18.136.157.135" not in content, f"Database host IP found in {doc.name}"

def test_readme_links_resolve():
    readme_path = PROJECT_ROOT / "README.md"
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find markdown relative file links: [text](path) where path does not start with http/https/#
    links = re.findall(r'\[.*?\]\((?!https?://|#)(.*?)\)', content)
    for link in links:
        # Strip query or anchors
        clean_link = link.split('#')[0].split('?')[0].strip()
        if not clean_link:
            continue
        target_path = (PROJECT_ROOT / clean_link).resolve()
        assert target_path.exists(), f"Broken relative link in README.md: '{clean_link}' does not exist on disk!"
