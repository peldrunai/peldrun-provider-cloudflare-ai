# Changelog

All notable changes to the Cloudflare Workers AI PELDRUN provider are
documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release section is prepended by `scripts/update_version.py`.
Contributors MUST fill in the placeholder bullets before tagging a
release — see CONTRIBUTING.md.

## [1.0.2] - 2026-11-08

### Fixed

- **README**: update badge links to point to the correct repository (f4079ec)

## [1.0.1] - 2026-11-08

### Fixed

- **ci**: extract single SHA-256 line; drop pytest-asyncio config (ee4822c)

## [Unreleased]

### Planned

- Support for `responses` API surface (GPT-OSS on Cloudflare).
- Full parallel tool calling (pending upstream confirmation).
- Streaming token-usage reporting in more models.
- Optional mTLS / custom CA bundle support (via `certificates` block).

## [1.0.0] - 2025-01-01

### Added

- Initial release of the Cloudflare Workers AI provider for PELDRUN.
- Native adapter (`CloudflareAIAdapter`) speaking the OpenAI-compatible
  surface at `/client/v4/accounts/{account_id}/ai/v1/chat/completions`.
- Dynamic base URL construction from `account_id` — no manual endpoint
  configuration required.
- Model discovery via the native `/ai/models/search` endpoint, merged
  with hardcoded facts for 13 models (Llama 3.1/3.2/3.3, Mistral,
  Qwen 2.5 Coder, QwQ 32B, DeepSeek R1 Distill, Kimi K2.5, Gemma 3).
- Streaming (SSE) support via the OpenAI SDK.
- Tool calling support for models that declare it (Llama 3.1/3.3,
  Qwen Coder, Kimi K2.5).
- Vision support for multimodal models (Llama 3.2 Vision, Mistral
  Small 3.1, Kimi K2.5, Gemma 3).
- Reasoning support for QwQ 32B, DeepSeek R1 Distill, Kimi K2.5.
- Structured output (`response_format`) support.
- Connection fields:
    - `account_id` (required, pattern-validated).
    - `api_key` (required, stored as `connection.api_key`).
    - `base_url` (optional, auto-computed).
- Runtime hints: `docs_url` and `requires_token_refresh: false`.
- Packaging scripts:
    - `scripts/build_package.py` — builds `dist/cloudflare-ai-<version>.zip`.
    - `scripts/verify_package.py` — deep ZIP verification.
    - `scripts/generate_catalog_entry.py` — produces catalog metadata.
    - `scripts/update_version.py` — synchronizes version across files.
- CI workflows:
    - `validate.yml` — runs on every PR.
    - `release.yml` — builds and uploads the ZIP + catalog entry on
      every `v*` tag.

### Known Limitations

- `max_tokens` is used instead of `max_completion_tokens`. The
  Cloudflare compatible endpoint does not yet support the latter.
- `parallel_tool_calls` is declared `false` — Cloudflare does not
  confirm support on the compatible endpoint.
- `forced_tool_choice` is declared `false` — same reason.
- `responses` API surface is not exposed via the adapter (GPT-OSS
  models would require it).

[Unreleased]: https://github.com/peldrun/peldrun-providers/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/peldrun/peldrun-providers/releases/tag/v1.0.0