"""Strict parsing and validation for LLM-induced provenance states."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

ENTITY_TYPES = frozenset(
    {
        "government_public_authority",
        "intergovernmental_org",
        "commercial_org",
        "academic_research",
        "fact_checker",
        "news_media",
        "reference_repository",
        "ngo_civil_society",
        "individual_or_user",
        "other_unknown",
    }
)
EVIDENCE_ORIGINS = frozenset(
    {
        "original_publication",
        "secondary_reporting",
        "rehosted_archive",
        "platform_hosted",
        "mixed_sources",
        "unclear",
    }
)
CLAIM_RELATIONS = frozenset(
    {
        "claim_subject_official",
        "designated_authority",
        "independent_secondary",
        "repository_only",
        "user_generated",
        "unclear",
    }
)


class PredictionValidationError(ValueError):
    """Raised when a model response violates the source-state contract."""


@dataclass(frozen=True)
class SourceStatePrediction:
    publisher_entity: str | None
    source_entity_type: str
    evidence_origin: str
    claim_relation: str
    confidence: float
    abstain: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_object(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise PredictionValidationError("response does not contain a JSON object")
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PredictionValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PredictionValidationError("top-level JSON value must be an object")
    return value


def validate_prediction(value: dict[str, Any]) -> SourceStatePrediction:
    required = {
        "publisher_entity",
        "source_entity_type",
        "evidence_origin",
        "claim_relation",
        "confidence",
        "abstain",
        "rationale",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise PredictionValidationError(f"missing fields: {missing}")
    publisher = value["publisher_entity"]
    if publisher is not None and (not isinstance(publisher, str) or not publisher.strip()):
        raise PredictionValidationError("publisher_entity must be a non-empty string or null")
    entity = value["source_entity_type"]
    origin = value["evidence_origin"]
    relation = value["claim_relation"]
    if isinstance(entity, list) and len(entity) == 1:
        entity = entity[0]
    if isinstance(origin, list) and len(origin) == 1:
        origin = origin[0]
    if isinstance(relation, list) and len(relation) == 1:
        relation = relation[0]
    if not isinstance(entity, str) or entity not in ENTITY_TYPES:
        raise PredictionValidationError(f"invalid source_entity_type: {entity}")
    if not isinstance(origin, str) or origin not in EVIDENCE_ORIGINS:
        raise PredictionValidationError(f"invalid evidence_origin: {origin}")
    if not isinstance(relation, str) or relation not in CLAIM_RELATIONS:
        raise PredictionValidationError(f"invalid claim_relation: {relation}")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise PredictionValidationError("confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise PredictionValidationError("confidence must be between 0 and 1")
    abstain = value["abstain"]
    if not isinstance(abstain, bool):
        raise PredictionValidationError("abstain must be boolean")
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise PredictionValidationError("rationale must be a non-empty string")
    if abstain and confidence > 0.5:
        raise PredictionValidationError("abstention requires confidence <= 0.5")
    return SourceStatePrediction(
        publisher_entity=publisher.strip() if isinstance(publisher, str) else None,
        source_entity_type=entity,
        evidence_origin=origin,
        claim_relation=relation,
        confidence=confidence,
        abstain=abstain,
        rationale=rationale.strip(),
    )


def parse_prediction(content: str) -> SourceStatePrediction:
    return validate_prediction(extract_json_object(content))
