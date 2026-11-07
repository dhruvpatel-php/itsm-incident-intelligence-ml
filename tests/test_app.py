import py_compile

from src.config import PROJECT_ROOT

def test_app_no_hardcoded_numbers():
    app_file = PROJECT_ROOT / "app.py"
    with open(app_file, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_hardcoded = [
        "80,021", "80021", "69.4", "0.8857", "225.2", "1331", "1,331", "0.5660", "18.136.157.135"
    ]
    for val in forbidden_hardcoded:
        assert val not in content, f"Hardcoded superseded value '{val}' found in app.py!"

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
