from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from datasketch import MinHash

TSHEG_PATTERN = re.compile(r"[་།]+")


def tibetan_syllables(text: str) -> list[str]:
    """Split text on tsheg/shad boundaries without normalizing."""
    return [part for part in TSHEG_PATTERN.split(text) if part.strip()]


def shingles_from_tokens(tokens: list[str], shingle_size: int) -> set[str]:
    if shingle_size <= 0:
        msg = "shingle_size must be positive"
        raise ValueError(msg)
    if len(tokens) < shingle_size:
        return {"_".join(tokens)} if tokens else set()
    return {
        "_".join(tokens[index : index + shingle_size])
        for index in range(len(tokens) - shingle_size + 1)
    }


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def minhash_values(items: Iterable[str], *, num_perm: int) -> tuple[int, ...]:
    minhash = MinHash(num_perm=num_perm)
    updated = False
    for item in items:
        minhash.update(item.encode("utf-8"))
        updated = True
    if not updated:
        return ()
    return tuple(int(value) for value in minhash.hashvalues)


def lsh_bands(hashvalues: tuple[int, ...], *, bands: int, rows: int) -> set[str]:
    if not hashvalues:
        return set()
    result: set[str] = set()
    for band_idx in range(bands):
        start = band_idx * rows
        band_slice = hashvalues[start : start + rows]
        if len(band_slice) != rows:
            continue
        digest = hashlib.md5(",".join(str(value) for value in band_slice).encode("ascii")).hexdigest()[:16]  # noqa: S324
        result.add(f"b{band_idx}_{digest}")
    return result


def cosine(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b, strict=True))

