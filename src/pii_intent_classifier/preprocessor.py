from typing import List, Dict, Union, Tuple


def preprocess_input(
    input_data: Union[str, List[str], List[Dict[str, str]]],
) -> Tuple[Union[str, List[str]], str, Union[int, None]]:
    """
    Normalizes input data for the classifier.

    Returns:
        tuple: (normalized_input, input_type, message_count)
    """
    if input_data is None:
        raise TypeError("Input cannot be None")

    if isinstance(input_data, str):
        return input_data, "string", None

    if not isinstance(input_data, list):
        raise TypeError(
            f"Input must be str, list[str], or list[dict], not {type(input_data).__name__}"
        )

    if not input_data:
        return [], "batch", 0

    if all(isinstance(item, str) for item in input_data):
        return input_data, "batch", None

    if all(isinstance(item, dict) for item in input_data):
        message_contents = []
        for i, msg in enumerate(input_data):
            if "content" not in msg:
                raise ValueError(
                    f"Message at index {i} is missing required 'content' key"
                )
            if not isinstance(msg["content"], str):
                raise TypeError(f"Message content at index {i} must be a string")
            message_contents.append(msg["content"])

        normalized_str = "\n".join(message_contents)
        return normalized_str, "message_array", len(input_data)

    raise TypeError("Input list must contain either all strings or all dictionaries")
