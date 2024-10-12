from typing import Any, Dict, Optional

from pydantic import BaseModel, PrivateAttr


class CacheHandler(BaseModel):
    """Cache for storing tool outputs."""

    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def exists(self, tool, input) -> bool:
        return f"{tool}-{input}" in self._cache

    def add(self, tool, input, output):
        self._cache[f"{tool}-{input}"] = output

    def read(self, tool, input) -> Optional[str]:
        return self._cache.get(f"{tool}-{input}")
