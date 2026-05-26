from __future__ import annotations

from iad.core.paths import (
    data_dir,
    exports_dir,
    logs_dir,
    models_dir,
    project_root,
    reports_dir,
    safe_filename,
)


def test_paths_resolve_to_existing_dirs() -> None:
    for fn in (project_root, data_dir, models_dir, logs_dir, reports_dir, exports_dir):
        path = fn()
        assert path.exists(), f"{fn.__name__} should exist"


def test_safe_filename_strips_unsafe_chars() -> None:
    # Path separators must be replaced (not preserved) — that's the security guarantee.
    sanitized = safe_filename("../../etc/passwd")
    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert sanitized == "passwd"

    assert safe_filename("model v1.0 (final).pkl") == "model_v1.0_(final).pkl"
    assert safe_filename("") == "file"


def test_safe_filename_blocks_path_traversal() -> None:
    """Returned name must never contain a directory separator."""
    for hostile in (
        "../../../etc/shadow",
        "..\\..\\windows\\system32\\config",
        "/abs/path/file.csv",
        "C:\\Users\\admin\\secret.xlsx",
    ):
        result = safe_filename(hostile)
        assert "/" not in result and "\\" not in result


def test_safe_filename_preserves_dotted_extensions() -> None:
    assert safe_filename("report.final.pdf").endswith(".pdf")
