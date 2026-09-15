"""Deterministic fingerprints for a sealed JAX compilation-cache tree."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CACHE_TREE_SCHEMA = b"trajectory-imf-jax-cache-tree-v2\0"


def _file_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def cache_tree_sha256(root: Path) -> str:
    """Hash every directory, file path, and file payload in a cache tree.

    The cache must contain at least one regular file. Symbolic links and other
    special entries are rejected so the digest cannot depend on targets outside
    the declared cache namespace.
    """

    if root.is_symlink():
        raise ValueError("cache root is a symbolic link")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("cache root is not a directory")
    entries = sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    )
    digest = hashlib.sha256()
    digest.update(_CACHE_TREE_SCHEMA)
    file_count = 0
    for path in entries:
        if path.is_symlink():
            raise ValueError("cache tree contains a symbolic link")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_dir():
            kind = b"D"
            payload = b""
        elif path.is_file():
            kind = b"F"
            payload = path.stat().st_size.to_bytes(8, "big") + _file_sha256(path)
            file_count += 1
        else:
            raise ValueError("cache tree contains an unsupported entry")
        digest.update(kind)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(payload)
    if file_count == 0:
        raise ValueError("cache root contains no files")
    return digest.hexdigest()
