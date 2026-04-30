def preprocess_input(
    input_data: str | list[str] | list[dict[str, str]],
) -> tuple[str | list[str], str, int | None]:
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

    first_item = input_data[0]

    if isinstance(first_item, str):
        str_list: list[str] = []
        for i, item in enumerate(input_data):
            if not isinstance(item, str):
                raise TypeError(f"Item at index {i} must be a string")
            str_list.append(item)
        return str_list, "batch", None

    if isinstance(first_item, dict):
        message_contents: list[str] = []
        for i, item in enumerate(input_data):
            if not isinstance(item, dict):
                raise TypeError(f"Item at index {i} must be a dictionary")

            msg = item
            if "content" not in msg:
                raise ValueError(
                    f"Message at index {i} is missing required 'content' key"
                )
            content = msg["content"]
            if not isinstance(content, str):
                raise TypeError(f"Message content at index {i} must be a string")
            message_contents.append(content)

        normalized_str = "\n".join(message_contents)
        return normalized_str, "message_array", len(input_data)

    raise TypeError("Input list must contain either all strings or all dictionaries")
