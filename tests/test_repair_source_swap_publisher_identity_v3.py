from __future__ import annotations

from provtimerag.data import EvidenceRecord
from scripts.prepare.repair_source_swap_publisher_identity_v3 import (
    canonicalize_evidence,
)


def evidence(source_url: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="e1",
        text="Evidence",
        source_id="web.archive.org",
        source_role="web",
        document_id="legacy-document",
        version_id="legacy-version",
        metadata={"source_url": source_url},
    )


def test_wayback_access_host_becomes_publisher_identity() -> None:
    item, audit = canonicalize_evidence(
        evidence(
            "https://web.archive.org/web/20240102030405/"
            "https://Publisher.Example/story?id=1"
        )
    )
    assert item.source_id == "publisher.example"
    assert item.metadata["source_url"] == "https://Publisher.Example/story?id=1"
    assert item.metadata["archive_access_url"].startswith("https://web.archive.org/")
    assert item.metadata["access_source_id"] == "web.archive.org"
    assert audit["resolved"] is True


def test_unresolved_wayback_identity_is_preserved_and_marked() -> None:
    item, audit = canonicalize_evidence(
        evidence("https://web.archive.org/unsupported")
    )
    assert item.source_id == "web.archive.org"
    assert item.metadata["publisher_identity_contract"] == "archive_unresolved_v3"
    assert audit["resolved"] is False


def test_direct_publisher_is_not_modified() -> None:
    original = evidence("https://publisher.example/story").model_copy(
        update={"source_id": "publisher.example"}
    )
    item, audit = canonicalize_evidence(original)
    assert item == original
    assert audit["archive_candidate"] is False
