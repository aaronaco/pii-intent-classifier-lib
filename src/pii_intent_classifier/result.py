from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClassificationResult:
    """Result of a PII intent classification."""

    is_flagged: bool
    flagged_category: List[str]
    asking_score: float
    giving_score: float
    combined_score: float
    input_type: str
    message_count: Optional[int]
    truncated: bool
