# src/cloudflare_ai_peldrun/__init__.py
"""
Cloudflare Workers AI provider adapter package for PELDRUN.

This package ships a single adapter — :class:`CloudflareAIAdapter` — that
teaches PELDRUN how to talk to Cloudflare Workers AI. It uses the
OpenAI-compatible surface at

    https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1

for chat completions, and the native REST surface at

    https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search

for model discovery, because the OpenAI SDK does not expose arbitrary
paths.

The package is loaded by PELDRUN's plugin host inside an isolated venv
created from ``requirements.txt``. It declares ``transport: "native"``
in the manifest, so every request crosses an IPC boundary before reaching
:meth:`CloudflareAIAdapter.generate` or
:meth:`CloudflareAIAdapter.stream`.

See ``manifest.json`` for the complete provider declaration and
``adapter.py`` for the implementation.
"""

from .adapter import CloudflareAIAdapter

__all__ = ["CloudflareAIAdapter"]
__version__ = "1.0.2"