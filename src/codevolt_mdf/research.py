"""Contracts for governed research discovery and dataset admission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .core import ContractError


class IntendedUse(str, Enum):
    TRAINING = "training"
    RETRIEVAL = "retrieval"
    EVALUATION = "evaluation"
    RESEARCH_ONLY = "research-only"


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SourceCandidate:
    source_id: str
    url: str
    retrieved_at: str
    publisher: str
    content_sha256: str
    rights_status: str
    privacy_review: str
    intended_use: IntendedUse
    relevance: float
    risks: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SourceCandidate:
        try:
            return cls(
                source_id=str(payload["source_id"]),
                url=str(payload["url"]),
                retrieved_at=str(payload["retrieved_at"]),
                publisher=str(payload["publisher"]),
                content_sha256=str(payload["content_sha256"]),
                rights_status=str(payload["rights_status"]),
                privacy_review=str(payload["privacy_review"]),
                intended_use=IntendedUse(payload["intended_use"]),
                relevance=float(payload["relevance"]),
                risks=tuple(str(risk) for risk in payload.get("risks", [])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"Invalid source candidate: {exc}") from exc

    def validate(self) -> None:
        required = (
            self.source_id,
            self.url,
            self.retrieved_at,
            self.publisher,
            self.content_sha256,
        )
        if not all(value.strip() for value in required):
            raise ContractError("Source identity, provenance, retrieval time, and hash are required")
        if not self.url.startswith(("https://", "http://")):
            raise ContractError("Source URL must use HTTP or HTTPS")
        if not 0 <= self.relevance <= 1:
            raise ContractError("Relevance must be between 0 and 1")

    def admission_decision(self) -> ReviewDecision:
        """Fail closed; discovery never grants training permission."""
        self.validate()
        if self.rights_status == "rejected" or self.privacy_review == "rejected":
            return ReviewDecision.REJECTED
        if self.rights_status != "approved" or self.privacy_review != "approved":
            return ReviewDecision.QUARANTINED
        return ReviewDecision.APPROVED

    def eligible_for(self, use: IntendedUse) -> bool:
        return self.admission_decision() == ReviewDecision.APPROVED and self.intended_use == use


@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str
    version: str
    content_sha256: str
    source_ids: tuple[str, ...]
    licence_review: str
    privacy_review: str
    contamination_check: str
    immutable: bool

    def validate_for_training(self) -> None:
        if not self.dataset_id or not self.version or not self.content_sha256:
            raise ContractError("Dataset identity, version, and content hash are required")
        if not self.source_ids:
            raise ContractError("A training dataset must identify its admitted sources")
        required_reviews = {
            "licence": self.licence_review,
            "privacy": self.privacy_review,
            "contamination": self.contamination_check,
        }
        incomplete = [name for name, status in required_reviews.items() if status != "passed"]
        if incomplete:
            raise ContractError(f"Dataset reviews are incomplete: {', '.join(incomplete)}")
        if not self.immutable:
            raise ContractError("A training dataset version must be immutable")
