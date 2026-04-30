from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Result of a PII intent classification."""

    is_flagged: bool
    flagged_category: list[str]
    asking_score: float
    giving_score: float
    combined_score: float
    input_type: str
    message_count: int | None
    truncated: bool
