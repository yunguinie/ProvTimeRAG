"""Query-safe entry point for the archive publisher identity audit."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlparse

from scripts.evaluate import audit_archive_source_identity as base


def wayback_target_url(url: Any) -> str | None:
    text = str(url or "").strip()
    parsed = urlparse(text)
    if (parsed.hostname or "").lower() not in base.ARCHIVE_HOSTS:
        return None
    decoded = unquote(parsed.path)
    if parsed.query:
        decoded = f"{decoded}?{parsed.query}"
    match = re.match(r"^/web/[^/]+/(https?://.+)$", decoded, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(https?://.+)$", decoded, flags=re.IGNORECASE)
    return match.group(1) if match else None


def recover_publisher_host(
    metadata: dict[str, Any],
) -> tuple[str | None, str | None]:
    for field in ("source_url", "cached_source_url", "swapped_source_url"):
        target = wayback_target_url(metadata.get(field))
        target_host = base.host(target)
        if target_host and target_host not in base.ARCHIVE_HOSTS:
            return target_host, field
    return None, None


def main() -> None:
    base.wayback_target_url = wayback_target_url
    base.recover_publisher_host = recover_publisher_host
    base.main()


if __name__ == "__main__":
    main()
