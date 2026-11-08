import ast
import re
import py_compile
from src.config import PROJECT_ROOT

WHITELIST_PATTERNS = [
    r'#[0-9a-fA-F]{3,6}',                     # Hex colors
    r'\b\d+(?:\.\d+)?(?:rem|px|em|pt)',       # CSS units
    r'\b\d+(?:\.\d+)?%',                      # Percentage formatting (e.g. 95%, 5.0%)
    r'\b(?:font-weight:\s*)?[1-9]00\b',       # CSS font weights (e.g. 700, 800)
    r'\bh[1-6]\b',                            # HTML tags h1-h6
    r'\bPillar\s+[1-4]\b',                    # Pillar names
    r'\bITIL(?:\s+V[1-4])?\b',                # ITIL framework
    r'\bP[1-5]\b',                            # Priority tokens P1-P5
    r'Oct\s+2013\s*-\s*Mar\s+2014\s+observed',# Requested observed label
    r'N\s*(?:>=|&ge;)\s*10',                  # Sample size threshold N >= 10
    r'm\s*=\s*10\.0',                         # Prior weight m=10.0
    r'at least 10 incidents',                 # N=10 rule description
    r'\b2D\b',                                # 2D ITIL lookup
    r'\b95%',                                 # 95% Confidence interval
]

def clean_whitelisted(text: str) -> str:
    for pat in WHITELIST_PATTERNS:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
    return text

def test_app_no_hardcoded_numbers():
    app_file = PROJECT_ROOT / "app.py"
    with open(app_file, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_hardcoded = [
        "80,021", "80021", "69.4", "0.8857", "225.2", "1331", "1,331", "0.5660", "18.136.157.135"
    ]
    for val in forbidden_hardcoded:
        assert val not in content, f"Hardcoded superseded value '{val}' found in app.py!"

def test_app_no_digit_literals_in_markdown_or_info():
    """
    Fail on any digit-containing literal in st.markdown/st.info/etc. strings
    (except whitelisted structural numbers).
    """
    app_file = PROJECT_ROOT / "app.py"
    source = app_file.read_text(encoding="utf-8")
    tree = ast.parse(source)

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in ('markdown', 'info', 'warning', 'error', 'success', 'caption'):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        cleaned = clean_whitelisted(arg.value)
                        digits = re.findall(r'\b\d+(?:\.\d+)?\b', cleaned)
                        if digits:
                            violations.append((node.lineno, func_name, digits, arg.value))
                    elif isinstance(arg, ast.JoinedStr):
                        for part in arg.values:
                            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                                cleaned = clean_whitelisted(part.value)
                                digits = re.findall(r'\b\d+(?:\.\d+)?\b', cleaned)
                                if digits:
                                    violations.append((node.lineno, func_name, digits, part.value))

    assert not violations, (
        "Found unwhitelisted digit-containing literal(s) in st.markdown/st.info:\n"
        + "\n".join(f"Line {v[0]} ({v[1]}): digits {v[2]} in '{v[3][:80]}'" for v in violations)
    )

def test_app_compiles():
    app_file = PROJECT_ROOT / "app.py"
    compiled = py_compile.compile(str(app_file), doraise=True)
    assert compiled is not None

def test_app_artifact_loading():
    # Directly test the load_artifacts logic from app.py
    import app
    p1, p2, p3_dept, p3_prio, p4_rfc, p4_assets, metrics = app.load_artifacts()
    assert p1 is not None
    assert p2 is not None
    assert p3_dept is not None
    assert p3_prio is not None
    assert p4_rfc is not None
    assert len(p4_assets) == 287
    assert metrics["pillar1_early_warning"]["test_roc_auc"] > 0.90
