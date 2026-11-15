"""Kiểm tra các module chưa triển khai luôn trống (chạy độc lập, không cần pytest).

Dùng khi:
  - Trước khi commit:  python scripts/check_pending_modules.py
  - Trong CI:          python scripts/check_pending_modules.py
  - Trong pre-commit hook

Exit code 0 = OK, exit code 1 = có vi phạm.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Module CHƯA triển khai: file .py tồn tại nhưng phải rỗng hoặc chỉ là placeholder.
PENDING_MODULES: dict[str, list[str]] = {
    "src/evaluation": [
        "__init__.py",
        "__main__.py",
        "metrics.py",
        "visualize.py",
        "visualize_html.py",
    ],
    "src/demo": [
        "__init__.py",
        "__main__.py",
    ],
}

# Module ĐÃ triển khai: file bắt buộc có code thật.
IMPLEMENTED_MODULES: dict[str, list[str]] = {
    "src/data": [
        "__main__.py",
        "audio_dataset.py",
        "dataloaders.py",
        "face_dataset.py",
        "make_processed.py",
        "make_splits.py",
        "split.py",
        "transforms.py",
    ],
    "src/models": [
        "__main__.py",
        "architecture.py",
        "audio_cnn.py",
        "face_cnn.py",
        "fusion.py",
    ],
    "src/training": [
        "__main__.py",
        "trainer.py",
    ],
}


def has_executable_code(path: Path) -> bool:
    """True nếu file chứa lệnh thực thi thật (không phải chỉ docstring/import/hằng số)."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign):
            if all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
                continue
        return True
    return False


def main() -> int:
    errors: list[str] = []

    for module, files in sorted(PENDING_MODULES.items()):
        module_dir = REPO_ROOT / module
        for filename in files:
            path = module_dir / filename
            if not path.exists():
                errors.append(
                    f"[MISSING] {path.relative_to(REPO_ROOT)} không tồn tại. "
                    f"Module chưa triển khai vẫn cần giữ file rỗng để thành viên có chỗ làm."
                )
                continue
            if has_executable_code(path):
                errors.append(
                    f"[PENDING] {path.relative_to(REPO_ROOT)} chứa code thực thi. "
                    f"Nếu module đã xong, cập nhật PENDING_MODULES trong file này "
                    f"+ tests/unit/test_pending_modules.py + docs/progress_status.md."
                )

    for module, files in sorted(IMPLEMENTED_MODULES.items()):
        module_dir = REPO_ROOT / module
        for filename in files:
            path = module_dir / filename
            rel = path.relative_to(REPO_ROOT)
            if not path.exists():
                errors.append(f"[MISSING] {rel} không tồn tại nhưng module {module} đã triển khai.")
            elif path.stat().st_size == 0:
                errors.append(f"[EMPTY] {rel} rỗng nhưng module {module} đã triển khai.")
            elif not has_executable_code(path):
                errors.append(f"[NO-CODE] {rel} không có lệnh thực thi nào.")

    protected = set(PENDING_MODULES)
    for py_file in (REPO_ROOT / "src").rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(f"{mod}/") for mod in protected):
            continue
        text = py_file.read_text(encoding="utf-8")
        for mod in protected:
            name = mod.split("/")[-1]
            if f"from {name}" in text or f"import {name}" in text:
                errors.append(f"[IMPORT] {rel} import tới module chưa triển khai '{name}'.")

    if errors:
        print("FAIL - vi pham trang thai module:")
        for err in errors:
            print(f"  - {err}")
        return 1

    pending_count = sum(len(f) for f in PENDING_MODULES.values())
    impl_count = sum(len(f) for f in IMPLEMENTED_MODULES.values())
    print(f"OK - {pending_count} file thuoc module chua trien khai deu trong/placeholder, "
          f"{impl_count} file thuoc module da trien khai deu co code.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
