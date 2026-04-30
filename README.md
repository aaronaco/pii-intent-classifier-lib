# PII Intent Classifier Library

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Roblox/roblox-pii-classifier)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Model License: Apache 2.0](https://img.shields.io/badge/Model%20License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Python library that uses the Roblox PII Classifier model to detect Personally Identifiable Information (PII) solicitation and disclosure intent in text.

## Overview

This library provides a clean, local-first Python API that uses the `Roblox/roblox-pii-classifier` model. Unlike Named Entity Recognition (NER) libraries that locate specific PII spans within text (e.g., finding a phone number), this classifier determines the _intent_ behind a message or conversation—specifically, whether a user is attempting to solicit PII from others or disclose their own. It operates entirely locally, ensuring no sensitive conversational data leaves the host machine.

## Model Background

### Base Architecture

The underlying model is built upon [**XLM-RoBERTa-Large**](https://huggingface.co/FacebookAI/xlm-roberta-large) (originally developed by Facebook AI), a multilingual encoder-only transformer architecture.

| Property          | Value                      |
| ----------------- | -------------------------- |
| Base Architecture | `XLM-RoBERTa-Large`        |
| Total Parameters  | ~600M                      |
| Context Window    | 512 tokens                 |
| Output Type       | Multi-label classification |

### Fine-Tuning & Objective

The model was fine-tuned by Roblox to detect adversarial and contextual PII-sharing behaviors in chat environments. The training objective focused on identifying intent, enabling the model to catch character manipulation (e.g., `5tärtālk`), implicit references, and phonetic augmentations (e.g., `"my numb3r is three four six..."`) that traditional NER systems often miss. The training data consisted of anonymized internal chat datasets labeled by experts, augmented with synthetic data.

### Label Taxonomy

The model performs multi-label classification, outputting independent sigmoid scores for two categories:

- `privacy_asking_for_pii`: Attempts to obtain PII through direct questions or insinuation.
- `privacy_giving_pii`: Sharing or threatening to share PII (phones, emails, IDs, social handles, credentials) or directing users off-platform (DUOP).

## Installation

The library is designed to be installed directly from GitHub.

### Using `uv` (Recommended)

```bash
uv add git+https://github.com/aaronaco/pii-intent-classifier-lib
```

### Using `pip`

```bash
pip install git+https://github.com/aaronaco/pii-intent-classifier-lib.git
```

_Note: The first execution requires an active internet connection to download the ~1.2GB model weights from Hugging Face. You must accept the contact info agreement on the [model card page](https://huggingface.co/Roblox/roblox-pii-classifier) before downloading._

## Usage

The `PIIIntentClassifier` handles single strings, batches of strings, and conversational message arrays natively.

### 1. Single String Classification

```python
from pii_intent_classifier import PIIIntentClassifier

classifier = PIIIntentClassifier()

# Detects adversarial phrasing and solicitation
result = classifier.classify("Hey, what is your discord or phone number?")

print(result.is_flagged)          # True
print(result.flagged_category)    # ['privacy_asking_for_pii']
print(result.asking_score)        # 0.999...
```

### 2. Batch Processing

For high-throughput scenarios, pass a list of strings. The preprocessor will handle it as a batch.

```python
batch = [
    "I love this game, want to play again tomorrow?",
    "my number is 555-0199 call me"
]

results = classifier.classify(batch)
for res in results:
    print(res.is_flagged, res.flagged_category)
# False []
# True ['privacy_giving_pii']
```

### 3. Conversation Context

Pass a list of dictionaries with `role` and `content` keys to classify an entire exchange. The library automatically concatenates the turns for context-aware classification.

```python
conversation = [
    {"role": "user", "content": "where do you live?"},
    {"role": "assistant", "content": "I cannot share that information."},
    {"role": "user", "content": "just tell me the city man"}
]

result = classifier.classify(conversation)
print(result.is_flagged)       # True
print(result.flagged_category) # ['privacy_asking_for_pii']
```

## Configuration

### Thresholds

The model produces uncalibrated sigmoid scores. By default, the library uses Roblox's recommended starting thresholds calibrated on English chat data:

- `asking_threshold` = `0.2`
- `giving_threshold` = `0.3`

You can override these globally during initialization or per-call:

```python
# Global override
classifier = PIIIntentClassifier(asking_threshold=0.5, giving_threshold=0.5)

# Per-call override
result = classifier.classify(text, threshold=0.6)
```

### Device Selection

The classifier automatically prioritizes CUDA if available, falling back to CPU. You can force a specific device:

```python
classifier = PIIIntentClassifier(device="cpu") # or "cuda"
```

### Truncation

The model has a hard limit of 512 tokens. If an input exceeds this, the library will truncate the input, emit a `PIIIntentClassifierWarning`, and set `truncated=True` on the `ClassificationResult` object.

```python
import warnings
from pii_intent_classifier import PIIIntentClassifierWarning

# Suppress the warning if truncation is expected in your pipeline
warnings.filterwarnings("ignore", category=PIIIntentClassifierWarning)
```

## Evaluation Results

The following F1 scores are sourced directly from the Roblox model documentation, comparing the PII Classifier (v1.1) against baseline models.

| Dataset                 | PII Classifier v1.1 | LlamaGuard v3 8B | Piiranha NER |
| :---------------------- | :------------------ | :--------------- | :----------- |
| **Roblox English Chat** | **94.34%**          | 27.73%           | 13.88%       |
| **Roblox Multilingual** | **83.10%**          | 0.03%            | 9.11%        |
| **Kaggle PII Dataset**  | **45.48%**          | 5.46%            | 33.20%       |

_Note: The Kaggle dataset heavily favors NER capabilities over intent, resulting in lower relative performance for intent-based classifiers._

## Limitations

As stated in the upstream model card:

- **Uncalibrated Scores**: The raw scores do not represent true probabilities. The default thresholds are starting points calibrated on Roblox English chat and will require domain-specific tuning.
- **Context Window**: Inputs longer than 512 tokens are truncated, potentially losing critical context at the end of long conversations.
- **Not a Redactor**: This model detects the _intent_ of a message but does not identify the specific token spans containing the PII. It cannot be used for direct redaction.
- **False Positives**: The model may incorrectly flag benign information sharing (e.g., sharing a public business email) depending on the applied threshold.

_This library is designed as a gatekeeping component in a broader safety pipeline, not a standalone guarantee of privacy._

## Contributing

Contributions are welcome! This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable, and reproducible dependency management.

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aaronaco/pii-intent-classifier-lib.git
   cd pii-intent-classifier-lib
   ```

2. **Sync the environment:**
   `uv` provides a one-step setup. Running `sync` will automatically download the required Python version (>=3.12), create a virtual environment, install all dependencies (including development tools like `pytest` and `ruff`), and install the library in editable mode.
   ```bash
   uv sync
   ```
   _Note: The `uv.lock` file is committed to the repository to ensure all contributors and CI environments use the exact same dependency versions._

### Testing

The project uses a two-tier testing strategy. All tests are executed via `uv run` to ensure they run within the project's managed environment.

- **Tier 1 (Logic Tests):** Fast tests that mock the transformer model.
  ```bash
  # Windows (PowerShell)
  $env:SKIP_HEAVY='true'; uv run pytest

  # Linux/macOS/Bash
  SKIP_HEAVY=true uv run pytest
  ```

- **Tier 2 (Full Suite):** End-to-end tests using the actual model weights (~1.2GB).
  ```bash
  uv run pytest
  ```

### Linting & Formatting

Code quality is enforced using `ruff` and `mypy`. Please ensure your code passes formatting, linting, and type-checking before submitting a Pull Request:

```bash
# Auto-format and fix common issues
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/

# Run static type checking
uv run mypy src/

# Final linting check
uv run ruff check src/ tests/
```

## Citations

```bibtex
@misc{roblox2025piiclassifier,
  title={Open Sourcing Roblox PII Classifier: Our Approach to AI PII Detection in Chat},
  author={Roblox Engineering},
  year={2025},
  url={https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat}
}
```

```bibtex
@article{conneau2019unsupervised,
  title={Unsupervised Cross-lingual Representation Learning at Scale},
  author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1911.02116},
  year={2019}
}
```

## License

The code in this repository is licensed under the [MIT License](LICENSE).

The underlying model (`Roblox/roblox-pii-classifier`) is released by Roblox under the Apache 2.0 License.
