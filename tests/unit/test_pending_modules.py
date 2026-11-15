"""Kiểm tra các module chưa triển khai luôn trống.

Mục đích: một file Python rỗng (0 byte hoặc chỉ whitespace/comment) trông
giống như "đã làm xong" nhưng thực tế chưa có gì cả. Test này chặn trường hợp đó.

Quy tắc:
  - Module chưa triển khai: file .py PHẢI tồn tại (để thành viên có chỗ làm)
    và phải rỗng hoặc chỉ chứa placeholder (docstring/comment) — tuyệt đối
    không có lệnh thực thi nào.
  - Module đã triển khai: file phải CÓ code thật (không rỗng).

Danh sách module chưa / đã triển khai lấy từ docs/progress_status.md.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Module CHƯA triển khai: file .py được phép rỗng, nhưng nếu có code thì
# phải là placeholder (chỉ docstring + import + comment), không có logic.
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

# Module ĐÃ triển khai: file bắt buộc phải có code thật.
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


def _has_executable_code(path: Path) -> bool:
    """True nếu file chứa lệnh thực thi thật (không phải chỉ docstring/import/comment)."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        # Bỏ qua docstring
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue
        # Bỏ qua import
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        # Bỏ qua __all__ = [...] và hằng số
        if isinstance(node, ast.Assign):
            if all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
                continue
        return True
    return False


def _is_placeholder_only(path: Path) -> bool:
    """True nếu file rỗng, hoặc chỉ chứa docstring/import/hằng số."""
    return not _has_executable_code(path)


@pytest.mark.parametrize("module,files", sorted(PENDING_MODULES.items()))
def test_pending_module_files_are_placeholder_only(module: str, files: list[str]):
    """Module chưa triển khai: file phải tồn tại và rỗng hoặc chỉ là placeholder."""
    module_dir = REPO_ROOT / module
    assert module_dir.is_dir(), f"Thiếu thư mục {module}"
    for filename in files:
        path = module_dir / filename
        assert path.exists(), (
            f"Thiếu file {path.relative_to(REPO_ROOT)}. "
            f"Module chưa triển khai vẫn cần giữ file rỗng để thành viên có chỗ làm."
        )
        assert _is_placeholder_only(path), (
            f"{path.relative_to(REPO_ROOT)} đang chứa code thực thi. "
            f"Nếu module đã triển khai xong, cập nhật PENDING_MODULES "
            f"trong tests/unit/test_pending_modules.py và docs/progress_status.md."
        )


@pytest.mark.parametrize("module,files", sorted(PENDING_MODULES.items()))
def test_pending_module_files_are_empty_bytes(module: str, files: list[str]):
    """Module chưa triển khai: file nên là file rỗng (0 byte) hoặc chỉ placeholder."""
    module_dir = REPO_ROOT / module
    for filename in files:
        path = module_dir / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        # File rỗng thuần hoặc chỉ gồm docstring/comment/import/hằng số.
        assert not _has_executable_code(path), (
            f"{path.relative_to(REPO_ROOT)} có lệnh thực thi — module chưa triển khai "
            f"phải để file rỗng."
        )
        assert len(text.strip()) == 0 or _is_placeholder_only(path), (
            f"{path.relative_to(REPO_ROOT)} không còn là file trống/placeholder."
        )


@pytest.mark.parametrize("module,files", sorted(IMPLEMENTED_MODULES.items()))
def test_implemented_module_files_have_code(module: str, files: list[str]):
    """Module đã triển khai: file bắt buộc có code thật, không được rỗng."""
    module_dir = REPO_ROOT / module
    assert module_dir.is_dir(), f"Thiếu thư mục {module}"
    for filename in files:
        path = module_dir / filename
        assert path.exists(), f"Thiếu file {path.relative_to(REPO_ROOT)}"
        assert path.stat().st_size > 0, (
            f"{path.relative_to(REPO_ROOT)} đang rỗng nhưng module "
            f"{module} đã được đánh dấu là đã triển khai."
        )
        assert _has_executable_code(path), (
            f"{path.relative_to(REPO_ROOT)} không có lệnh thực thi nào."
        )


def test_pending_modules_have_no_stale_imports():
    """Không module nào khác được import tới module chưa triển khai."""
    protected = set(PENDING_MODULES)
    offenders: list[str] = []
    for py_file in (REPO_ROOT / "src").rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(f"{mod}/") for mod in protected):
            continue
        text = py_file.read_text(encoding="utf-8")
        for mod in protected:
            module_name = mod.split("/")[-1]
            if f"from {module_name}" in text or f"import {module_name}" in text:
                offenders.append(f"{rel} -> {module_name}")
    assert not offenders, (
        "Các file sau import tới module chưa triển khai: " + ", ".join(offenders)
    )
