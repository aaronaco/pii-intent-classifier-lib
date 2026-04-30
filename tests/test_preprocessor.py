import pytest
from pii_intent_classifier.preprocessor import preprocess_input


def test_preprocess_string():
    text = "Hello world"
    normalized, input_type, count = preprocess_input(text)
    assert normalized == text
    assert input_type == "string"
    assert count is None


def test_preprocess_batch():
    batch = ["msg1", "msg2"]
    normalized, input_type, count = preprocess_input(batch)
    assert normalized == batch
    assert input_type == "batch"
    assert count is None


def test_preprocess_conversation():
    conv = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    normalized, input_type, count = preprocess_input(conv)
    assert normalized == "hi\nhello"
    assert input_type == "message_array"
    assert count == 2


def test_preprocess_empty_list():
    normalized, input_type, count = preprocess_input([])
    assert normalized == []
    assert input_type == "batch"
    assert count == 0


def test_preprocess_errors():
    with pytest.raises(TypeError, match="Input cannot be None"):
        preprocess_input(None)

    with pytest.raises(TypeError, match="Input must be str"):
        preprocess_input(123)

    with pytest.raises(ValueError, match="missing required 'content' key"):
        preprocess_input([{"role": "user"}])

    with pytest.raises(TypeError, match="Message content at index 0 must be a string"):
        preprocess_input([{"content": 123}])

    with pytest.raises(
        TypeError,
        match="Item at index 1 must be a string",
    ):
        preprocess_input(["string", {"content": "dict"}])
