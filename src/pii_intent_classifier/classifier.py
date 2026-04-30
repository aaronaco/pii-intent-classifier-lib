import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any

from .result import ClassificationResult
from .preprocessor import preprocess_input
from .exceptions import PIIIntentClassifierWarning, PIIIntentClassifierError


class PIIIntentClassifier:
    """
    PII Intent Classifier library wrapping the Roblox PII Classifier model.
    """

    MODEL_ID = "Roblox/roblox-pii-classifier"
    MAX_TOKENS = 512
    DEFAULT_ASKING_THRESHOLD = 0.2
    DEFAULT_GIVING_THRESHOLD = 0.3
    LABEL_ASKING_PII = "privacy_asking_for_pii"
    LABEL_GIVING_PII = "privacy_giving_pii"

    def __init__(
        self,
        device: str = "auto",
        threshold: float | None = None,
        asking_threshold: float = DEFAULT_ASKING_THRESHOLD,
        giving_threshold: float = DEFAULT_GIVING_THRESHOLD,
    ):
        """
        Initialize the classifier.

        Args:
            device: "cpu", "cuda", or "auto".
            threshold: Global threshold override for both categories.
            asking_threshold: Threshold for privacy_asking_for_pii.
            giving_threshold: Threshold for privacy_giving_pii.
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.default_asking_threshold = (
            threshold if threshold is not None else asking_threshold
        )
        self.default_giving_threshold = (
            threshold if threshold is not None else giving_threshold
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_ID
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise PIIIntentClassifierError(
                f"Failed to load Roblox PII Classifier model from Hugging Face: {str(e)}\n"
                "Likely causes:\n"
                "1. Hugging Face contact info agreement not accepted. Visit: https://huggingface.co/Roblox/roblox-pii-classifier\n"
                "2. No internet connection.\n"
                "3. Insufficient disk space or memory."
            ) from e

        self.label_map = self.model.config.id2label
        self.asking_idx = next(
            k for k, v in self.label_map.items() if self.LABEL_ASKING_PII in v
        )
        self.giving_idx = next(
            k for k, v in self.label_map.items() if self.LABEL_GIVING_PII in v
        )

    def classify(
        self,
        input_data: str | list[str] | list[dict[str, str]],
        threshold: float | None = None,
        asking_threshold: float | None = None,
        giving_threshold: float | None = None,
    ) -> ClassificationResult | list[ClassificationResult]:
        """
        Classifies the input text for PII intent.

        Args:
            input_data: Single string, batch of strings, or list of message dicts.
            threshold: Global threshold override for this call.
            asking_threshold: Per-category override for this call.
            giving_threshold: Per-category override for this call.
        """
        if isinstance(input_data, list) and not input_data:
            return []

        normalized_input, input_type, message_count = preprocess_input(input_data)

        call_asking_threshold = (
            threshold
            if threshold is not None
            else (
                asking_threshold
                if asking_threshold is not None
                else self.default_asking_threshold
            )
        )
        call_giving_threshold = (
            threshold
            if threshold is not None
            else (
                giving_threshold
                if giving_threshold is not None
                else self.default_giving_threshold
            )
        )

        if input_type == "batch":
            # normalized_input is list[str] when input_type is "batch"
            assert isinstance(normalized_input, list)
            results = []
            for text in normalized_input:
                results.append(
                    self._run_inference(
                        text,
                        "batch",
                        None,
                        call_asking_threshold,
                        call_giving_threshold,
                    )
                )
            return results

        # normalized_input is str when input_type is not "batch"
        assert isinstance(normalized_input, str)
        return self._run_inference(
            normalized_input,
            input_type,
            message_count,
            call_asking_threshold,
            call_giving_threshold,
        )

    def is_flagged(
        self, input_data: str | list[str] | list[dict[str, str]], **threshold_kwargs: Any
    ) -> bool | list[bool]:
        """Convenience wrapper around classify() returning bool(s)."""
        result = self.classify(input_data, **threshold_kwargs)
        if isinstance(result, list):
            return [r.is_flagged for r in result]
        return result.is_flagged

    def _run_inference(
        self,
        text: str,
        input_type: str,
        message_count: int | None,
        asking_threshold: float,
        giving_threshold: float,
    ) -> ClassificationResult:
        if not text.strip():
            return ClassificationResult(
                is_flagged=False,
                flagged_category=[],
                asking_score=0.0,
                giving_score=0.0,
                combined_score=0.0,
                input_type=input_type,
                message_count=message_count,
                truncated=False,
            )

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )

        input_ids = tokens["input_ids"]
        token_count = input_ids.shape[1]
        truncated = False

        if token_count > self.MAX_TOKENS:
            truncated = True
            warnings.warn(
                f"Input exceeds {self.MAX_TOKENS} tokens (found {token_count}). It has been truncated.",
                PIIIntentClassifierWarning,
            )
            input_ids = input_ids[:, : self.MAX_TOKENS]
            attention_mask = tokens["attention_mask"][:, : self.MAX_TOKENS]
        else:
            attention_mask = tokens["attention_mask"]

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            scores = torch.sigmoid(logits).cpu().numpy()[0]

        asking_score = float(scores[self.asking_idx])
        giving_score = float(scores[self.giving_idx])
        combined_score = max(asking_score, giving_score)

        flagged_category = []
        if asking_score >= asking_threshold:
            flagged_category.append(self.LABEL_ASKING_PII)
        if giving_score >= giving_threshold:
            flagged_category.append(self.LABEL_GIVING_PII)

        return ClassificationResult(
            is_flagged=len(flagged_category) > 0,
            flagged_category=flagged_category,
            asking_score=asking_score,
            giving_score=giving_score,
            combined_score=combined_score,
            input_type=input_type,
            message_count=message_count,
            truncated=truncated,
        )
