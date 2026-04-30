import pytest
from unittest.mock import MagicMock, patch
from pii_intent_classifier import (
    PIIIntentClassifier,
    PIIIntentClassifierWarning,
    ClassificationResult,
)


@pytest.fixture
def mock_classifier():
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model,
    ):
        # Mock tokenizer
        mock_tokenizer_inst = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_inst

        # Mock model
        mock_model_inst = MagicMock()
        mock_model_inst.config.id2label = {
            0: "privacy_asking_for_pii",
            1: "privacy_giving_pii",
        }
        mock_model_inst.device = "cpu"
        mock_model.return_value = mock_model_inst

        classifier = PIIIntentClassifier(device="cpu")
        return classifier, mock_tokenizer_inst, mock_model_inst


def test_classifier_logic_single(mock_classifier):
    classifier, mock_tok, mock_mod = mock_classifier

    # Setup mock tokenizer output
    mock_tok.return_value = {
        "input_ids": MagicMock(shape=(1, 10)),
        "attention_mask": MagicMock(),
    }
    mock_tok.return_value["input_ids"].to.return_value = mock_tok.return_value[
        "input_ids"
    ]
    mock_tok.return_value["attention_mask"].to.return_value = mock_tok.return_value[
        "attention_mask"
    ]

    # Setup mock model output (logits)
    # simulate specific scores. scores = sigmoid(logits)
    # sigmoid(0) = 0.5
    # sigmoid(-2) ~= 0.12
    # sigmoid(2) ~= 0.88
    mock_logits = MagicMock()
    mock_logits.logits = MagicMock()

    mock_mod.return_value = mock_logits

    with patch("torch.sigmoid") as mock_sig:
        import numpy as np

        mock_sig.return_value.cpu.return_value.numpy.return_value = np.array(
            [[0.5, 0.1]]
        )

        res = classifier.classify("test message")

        assert isinstance(res, ClassificationResult)
        assert res.asking_score == 0.5
        assert res.giving_score == 0.1
        assert res.is_flagged is True
        assert res.flagged_category == ["privacy_asking_for_pii"]


def test_threshold_overrides(mock_classifier):
    classifier, mock_tok, mock_mod = mock_classifier

    mock_tok.return_value = {
        "input_ids": MagicMock(shape=(1, 10)),
        "attention_mask": MagicMock(),
    }

    with patch("torch.sigmoid") as mock_sig:
        import numpy as np

        mock_sig.return_value.cpu.return_value.numpy.return_value = np.array(
            [[0.25, 0.25]]
        )

        res = classifier.classify("test")
        assert res.is_flagged is True
        assert "privacy_asking_for_pii" in res.flagged_category

        res2 = classifier.classify("test", threshold=0.5)
        assert res2.is_flagged is False

        res3 = classifier.classify("test", asking_threshold=0.3)
        assert res3.is_flagged is False


def test_truncation_warning(mock_classifier):
    classifier, mock_tok, mock_mod = mock_classifier

    # Simulate > 512 tokens
    mock_tok.return_value = {
        "input_ids": MagicMock(shape=(1, 600)),
        "attention_mask": MagicMock(),
    }

    with patch("torch.sigmoid") as mock_sig:
        import numpy as np

        mock_sig.return_value.cpu.return_value.numpy.return_value = np.array(
            [[0.1, 0.1]]
        )

        with pytest.warns(PIIIntentClassifierWarning, match="Input exceeds 512 tokens"):
            res = classifier.classify("long text " * 100)
            assert res.truncated is True


def test_empty_inputs(mock_classifier):
    classifier, _, _ = mock_classifier

    # Empty string should return 0.0 scores
    res_str = classifier.classify("")
    assert res_str.is_flagged is False
    assert res_str.asking_score == 0.0
    assert res_str.giving_score == 0.0

    # Whitespace only should also return 0.0
    res_space = classifier.classify("   ")
    assert res_space.is_flagged is False
    assert res_space.asking_score == 0.0

    # Empty list should return empty list
    res_list = classifier.classify([])
    assert res_list == []
