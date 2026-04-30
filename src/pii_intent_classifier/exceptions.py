class PIIIntentClassifierWarning(UserWarning):
    """Warning emitted when input text is truncated to fit the model's token limit."""

    pass


class PIIIntentClassifierError(Exception):
    """Exception raised for errors in the PIIIntentClassifier library."""

    pass
