from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from scripts.dedup.methods.base import DedupMethod

MethodFactory = Callable[[dict[str, Any]], DedupMethod[Any]]

_REGISTRY: dict[str, MethodFactory] = {}


def register_method(name: str) -> Callable[[MethodFactory], MethodFactory]:
    def decorator(factory: MethodFactory) -> MethodFactory:
        _REGISTRY[name] = factory
        return factory

    return decorator


def create_method(name: str, options: dict[str, Any] | None = None) -> DedupMethod[Any]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        msg = f"Unknown dedup method {name!r}. Available: {available}"
        raise KeyError(msg)
    return _REGISTRY[name](options or {})


def available_methods() -> list[str]:
    return sorted(_REGISTRY)


def canonical_options(options: dict[str, Any]) -> str:
    return json.dumps(options, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def options_hash(options: dict[str, Any]) -> str:
    return hashlib.sha1(canonical_options(options).encode("utf-8")).hexdigest()[:10]  # noqa: S324


def load_builtin_methods() -> None:
    """Import modules for their registration side effects."""
    from scripts.dedup.methods import chunked_embedding as _chunked_embedding  # noqa: F401, PLC0415
    from scripts.dedup.methods import chunked_minhash as _chunked_minhash  # noqa: F401, PLC0415
    from scripts.dedup.methods import fasttext_embedding as _fasttext_embedding  # noqa: F401, PLC0415
    from scripts.dedup.methods import minhash_datasketch as _minhash_datasketch  # noqa: F401, PLC0415
    from scripts.dedup.methods import minhash_os_jaccard as _minhash_os_jaccard  # noqa: F401, PLC0415
    from scripts.dedup.methods import minhash_os_query as _minhash_os_query  # noqa: F401, PLC0415
    from scripts.dedup.methods import minhash_os_sidecar as _minhash_os_sidecar  # noqa: F401, PLC0415

