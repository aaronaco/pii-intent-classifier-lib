# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-30

### Added
- Initial release of the `pii-intent-classifier-lib` Python library.
- Core `PIIIntentClassifier` class providing a local-first wrapper around the `Roblox/roblox-pii-classifier` model.
- `classify()` method supporting three distinct input modes:
  - Single message classification (`str`).
  - Batch message classification (`list[str]`).
  - Context-aware conversation classification (`list[dict]` with role/content).
- `is_flagged()` convenience wrapper for quick boolean checks.
- `ClassificationResult` dataclass returning comprehensive inference metrics, including independent and combined scores, flagged categories, and input metadata.
- Configurable thresholds with defaults based on upstream model recommendations (`0.2` for asking, `0.3` for giving), capable of global or per-category overrides per call.
- Automatic device detection (CUDA or CPU) with manual override capabilities.
- Intelligent token truncation for inputs exceeding the 512-token context window, emitting a suppressable `PIIIntentClassifierWarning` and setting a `truncated=True` flag on the result object.
- Robust error handling featuring custom exceptions (`PIIIntentClassifierError`) and proactive input validation.
- End-to-end two-tier test suite using `pytest`, featuring fast logic verification (`SKIP_HEAVY=true`) and heavy full-model inference tests.
- Strict project tooling and dependency management via `uv`, alongside code quality enforcement using `ruff` and `mypy`.
