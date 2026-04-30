import pytest
import os
from pii_intent_classifier import PIIIntentClassifier


@pytest.mark.skipif(
    os.getenv("SKIP_HEAVY") == "true", reason="Skipping heavy model tests"
)
def test_classifier_full_scenarios():
    """
    End-to-end tests with the real model.
    Requires internet for first run and HF agreement.
    """
    try:
        classifier = PIIIntentClassifier()
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")

    # Asking only flagged (Direct question)
    res1 = classifier.classify("hey what's your discord? add me")
    assert res1.is_flagged is True
    assert "privacy_asking_for_pii" in res1.flagged_category

    # Adversarial pattern from research
    res2 = classifier.classify("my numb3r is actually three four six 555 zero 123")
    assert res2.is_flagged is True
    assert "privacy_giving_pii" in res2.flagged_category

    # Both flagged (Solicitation + Disclosure)
    res3 = classifier.classify(
        "tell me your address and i'll send you a gift. mine is 123 Maple St"
    )
    assert res3.is_flagged is True

    # Clean text (Contextual but safe)
    res4 = classifier.classify("I love playing this game with you, it's so much fun!")
    assert res4.is_flagged is False
    assert res4.flagged_category == []
