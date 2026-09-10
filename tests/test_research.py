import pytest

from codevolt_mdf.core import ContractError
from codevolt_mdf.research import (
    DatasetVersion,
    IntendedUse,
    ReviewDecision,
    SourceCandidate,
)


def candidate(rights="unknown", privacy="pending", use="research-only"):
    return SourceCandidate.from_dict(
        {
            "source_id": "source-1",
            "url": "https://example.com/source",
            "retrieved_at": "2026-09-09T17:00:00Z",
            "publisher": "Example",
            "content_sha256": "abc123",
            "rights_status": rights,
            "privacy_review": privacy,
            "intended_use": use,
            "relevance": 0.8,
            "risks": [],
        }
    )


def test_discovered_source_is_quarantined_by_default():
    assert candidate().admission_decision() == ReviewDecision.QUARANTINED


def test_approval_for_research_does_not_grant_training_use():
    source = candidate(rights="approved", privacy="approved")
    assert source.admission_decision() == ReviewDecision.APPROVED
    assert not source.eligible_for(IntendedUse.TRAINING)


def test_explicitly_approved_training_source_is_eligible():
    source = candidate(rights="approved", privacy="approved", use="training")
    assert source.eligible_for(IntendedUse.TRAINING)


def test_dataset_fails_closed_when_review_is_incomplete():
    dataset = DatasetVersion(
        dataset_id="dataset-1",
        version="1.0.0",
        content_sha256="abc123",
        source_ids=("source-1",),
        licence_review="passed",
        privacy_review="pending",
        contamination_check="passed",
        immutable=True,
    )
    with pytest.raises(ContractError, match="privacy"):
        dataset.validate_for_training()


def test_complete_immutable_dataset_is_valid_for_training():
    dataset = DatasetVersion(
        dataset_id="dataset-1",
        version="1.0.0",
        content_sha256="abc123",
        source_ids=("source-1",),
        licence_review="passed",
        privacy_review="passed",
        contamination_check="passed",
        immutable=True,
    )
    dataset.validate_for_training()
