# src/cloudflare_ai_peldrun/adapter.py
"""
Cloudflare Workers AI adapter for PELDRUN.

Talks to Cloudflare's OpenAI-compatible surface:

    POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions

for chat completions, and to the native REST surface:

    GET  https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search

for model discovery (the OpenAI SDK does not expose arbitrary paths).

Design invariants
-----------------

1. **Dynamic base URL.** The account ID is part of the URL path, so the
   adapter *computes* the base URL from ``connection.extra_params["account_id"]``
   at construction time. A user-supplied ``base_url`` is ignored with a
   warning — the manifest documents this.

2. **Uses AsyncOpenAI as the wire client.** Cloudflare officially supports
   the OpenAI Chat Completions protocol on ``/ai/v1/``, so the standard
   ``openai`` SDK works unmodified. We never touch raw HTTP for chat.

3. **Uses httpx only for model discovery.** ``/ai/models/search`` is a
   Cloudflare-specific endpoint not exposed by the OpenAI SDK.

4. **13 models have hardcoded facts.** ``/ai/models/search`` does not always
   return context length. We merge ``model_facts`` from ``manifest.json``
   with the live list, preferring the hardcoded values.

5. **Errors are canonical.** Every SDK exception is mapped to a
   ``ProviderError`` subclass. Cancellation is re-raised untouched.

6. **Streaming is truthful.** We request SSE from ``/ai/v1/chat/completions``
   with ``stream=True``. The OpenAI SDK parses it into delta chunks that we
   re-emit as ``LLMStreamEvent``. No buffering.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import openai
from openai import AsyncOpenAI

from peldrun.llm.client import (
    LLMResponse,
    TokenUsage,
    ToolCall,
    UsageSource,
)
from peldrun.llm.contracts import (
    AdapterFeatures,
    AdapterProfile,
    ConnectionContext,
    LLMStreamEvent,
    StreamEventType,
    ToolCallStyle,
)
from peldrun.llm.errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from peldrun.llm.request import LLMRequest
from peldrun.llm.request_compiler import CompiledRequest, compile_request


logger = logging.getLogger("cloudflare_ai_peldrun.adapter")


# A Cloudflare Account ID is a 32-character lowercase hexadecimal string.
# Validated here even though the manifest's `pattern` also enforces it,
# because the `pattern` is client-side only (documented in
# docs/providers/05-connection-fields.md §"pattern — for security").
_ACCOUNT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


# The masked placeholder that the Store UI displays for password fields
# when the user has not edited them. Should never reach the adapter, but
# we defensively reject it (documented in
# docs/providers/10-connection-management.md §"Secret masking").
_MASKED_PLACEHOLDER = "\u2022" * 8  # "••••••••"


_PROFILE = AdapterProfile(
    provider_id="cloudflare-ai",
    api_surface="chat_completions",
    features=AdapterFeatures(
        model_discovery=True,
        native_streaming=True,
        buffered_streaming=True,
        tool_calling=True,
        parallel_tool_calls=False,
        forced_tool_choice=False,
        vision=True,
        audio=False,
        structured_output=True,
        usage_reporting=True,
        reasoning=True,
    ),
    tool_call_style=ToolCallStyle.OPENAI_FUNCTION,
)


class CloudflareAIAdapter:
    """Native adapter for Cloudflare Workers AI."""

    ADAPTER_STYLE = "v1"
    API_CONTRACT = "peldrun.llm.provider.v1"
    PROFILE = _PROFILE
    provider_id = "cloudflare-ai"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        context: Any = None,
        *,
        connection: Optional[ConnectionContext] = None,
        profile: Optional[AdapterProfile] = None,
    ) -> None:
        # Support both constructor styles used by ProviderFactory
        # (mirrors openai_native.py).
        if context is not None:
            self._connection: ConnectionContext = context.connection
            self._profile: AdapterProfile = context.profile or _PROFILE
            self._credentials: Dict[str, str] = dict(context.credentials or {})
        else:
            if connection is None:
                raise TypeError(
                    "CloudflareAIAdapter requires either `context=` or "
                    "`connection=` (and optionally `profile=`)."
                )
            self._connection = connection
            self._profile = profile or _PROFILE
            self._credentials = (
                {"api_key": connection.api_key}
                if connection.api_key
                else {}
            )

        self.profile = self._profile

        # Load model_facts from the package's manifest.json.
        self._model_facts = self._load_model_facts()

        # Extract connection-specific fields.
        extra = dict(self._connection.extra_params or {})
        extra.pop("_peldrun_plugin", None)

        account_id = str(extra.get("account_id") or "").strip()
        if not account_id:
            raise ProviderAuthError(
                "Cloudflare Workers AI requires an 'account_id' field. "
                "Find it via `wrangler whoami` or in the Cloudflare dashboard URL.",
                provider=self.provider_id,
            )
        if not _ACCOUNT_ID_PATTERN.match(account_id):
            raise ProviderInvalidRequestError(
                f"account_id {account_id!r} is not a valid Cloudflare Account ID. "
                "It must be a 32-character lowercase hexadecimal string.",
                provider=self.provider_id,
            )

        # The manifest declares `api_key` (label "API Token"), so the
        # runtime hoists it to `connection.api_key`. We also accept it
        # in `extra` for defensive compatibility.
        api_key = (
            self._credentials.get("api_key")
            or self._connection.api_key
            or extra.get("api_key")
            or ""
        ).strip()

        if not api_key:
            raise ProviderAuthError(
                "Cloudflare Workers AI requires an 'api_key' field (API Token). "
                "Create one with the 'Workers AI Read' permission at "
                "dash.cloudflare.com/profile/api-tokens.",
                provider=self.provider_id,
            )
        if api_key == _MASKED_PLACEHOLDER:
            raise ProviderAuthError(
                "The API Token appears to be masked. Please re-enter it.",
                provider=self.provider_id,
            )

        # The manifest says base_url is auto-computed. Warn if the user
        # supplied something, but ignore it to guarantee a valid endpoint.
        custom_base = (self._connection.base_url or "").strip()
        if custom_base:
            logger.warning(
                "Custom base_url %r ignored. The Cloudflare AI adapter "
                "always computes the endpoint from account_id.",
                custom_base,
            )

        # Build the dynamic endpoints.
        self._api_base = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        )
        self._models_base = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai"
        )

        self._timeout = float(self._connection.timeout or 120.0)

        # OpenAI-compatible client for chat completions.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self._api_base,
            timeout=self._timeout,
            max_retries=0,  # PELDRUN's runtime handles retries.
        )

        # httpx client for the Cloudflare-specific model search endpoint.
        self._http = httpx.AsyncClient(
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )

        logger.debug(
            "CloudflareAIAdapter ready: account_id=%s... endpoint=%s",
            account_id[:8],
            self._api_base,
        )

    # ------------------------------------------------------------------
    # ProviderAdapter: generate (non-streaming)
    # ------------------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute a non-streaming chat completion."""
        started_at = time.time()
        compiled = compile_request(request, self._profile)

        try:
            response = await self._client.chat.completions.create(
                **self._build_chat_kwargs(compiled, stream=False),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _map_openai_exception(
                exc, provider=self.provider_id, model=compiled.model
            ) from exc

        result = self._convert_response(response, compiled.model)
        result.latency_ms = round((time.time() - started_at) * 1000.0, 2)
        result.ttft_ms = result.latency_ms
        return result

    # ------------------------------------------------------------------
    # ProviderAdapter: stream
    # ------------------------------------------------------------------

    async def stream(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        """Execute a streaming chat completion."""
        compiled = compile_request(request, self._profile)
        seq = 0

        try:
            stream = await self._client.chat.completions.create(
                **self._build_chat_kwargs(compiled, stream=True),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            seq += 1
            yield _error_event(
                _map_openai_exception(
                    exc, provider=self.provider_id, model=compiled.model
                ),
                sequence=seq,
            )
            return

        active_tools: Dict[int, Dict[str, Any]] = {}
        captured_usage: Optional[TokenUsage] = None
        finish_reason: Optional[str] = None

        try:
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    converted = _convert_usage(chunk_usage)
                    if converted is not None:
                        captured_usage = converted
                        seq += 1
                        yield LLMStreamEvent(
                            type=StreamEventType.USAGE_UPDATE,
                            usage=captured_usage,
                            sequence=seq,
                        )

                if not choices:
                    continue

                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    # Content token
                    content_piece = getattr(delta, "content", None)
                    if isinstance(content_piece, str) and content_piece:
                        seq += 1
                        yield LLMStreamEvent(
                            type=StreamEventType.CONTENT_DELTA,
                            content_delta=content_piece,
                            sequence=seq,
                        )

                    # Tool call deltas
                    raw_tc_deltas = getattr(delta, "tool_calls", None) or []
                    for tc_delta in raw_tc_deltas:
                        tool_index = getattr(tc_delta, "index", 0) or 0
                        fn = getattr(tc_delta, "function", None)

                        if tool_index not in active_tools:
                            tool_id = (
                                getattr(tc_delta, "id", None)
                                or f"call_{tool_index}"
                            )
                            tool_name = (
                                getattr(fn, "name", None) if fn else None
                            ) or ""
                            active_tools[tool_index] = {
                                "id": tool_id,
                                "name": tool_name,
                                "arguments": "",
                            }
                            seq += 1
                            yield LLMStreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_call_id=tool_id,
                                tool_name=tool_name,
                                tool_index=tool_index,
                                sequence=seq,
                            )
                        else:
                            late_id = getattr(tc_delta, "id", None)
                            if late_id:
                                active_tools[tool_index]["id"] = late_id

                        if fn is not None:
                            args_piece = getattr(fn, "arguments", None)
                            if isinstance(args_piece, str) and args_piece:
                                active_tools[tool_index]["arguments"] += args_piece
                                seq += 1
                                yield LLMStreamEvent(
                                    type=StreamEventType.TOOL_CALL_DELTA,
                                    tool_call_id=active_tools[tool_index]["id"],
                                    tool_name=active_tools[tool_index]["name"],
                                    tool_index=tool_index,
                                    tool_arguments_delta=args_piece,
                                    sequence=seq,
                                )

                chunk_finish = getattr(choice, "finish_reason", None)
                if chunk_finish:
                    finish_reason = chunk_finish

        except asyncio.CancelledError:
            try:
                await stream.close()
            except Exception:
                pass
            raise
        except Exception as exc:
            seq += 1
            yield _error_event(
                _map_openai_exception(
                    exc, provider=self.provider_id, model=compiled.model
                ),
                sequence=seq,
            )
            return

        # Emit TOOL_CALL_COMPLETE for each accumulated tool.
        for tool_idx in sorted(active_tools.keys()):
            tool = active_tools[tool_idx]
            parsed_args: Any = tool["arguments"]
            if isinstance(parsed_args, str) and parsed_args:
                try:
                    parsed_args = json.loads(parsed_args)
                except Exception:
                    parsed_args = {"raw": parsed_args}
            elif not parsed_args:
                parsed_args = {}
            seq += 1
            yield LLMStreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call_id=tool["id"],
                tool_name=tool["name"],
                tool_index=tool_idx,
                tool_arguments_complete=parsed_args,
                sequence=seq,
            )

        seq += 1
        yield LLMStreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason or "stop",
            usage=captured_usage,
            sequence=seq,
        )

    # ------------------------------------------------------------------
    # ProviderAdapter: discover_models
    # ------------------------------------------------------------------

    async def discover_models(
        self, connection: Optional[ConnectionContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Query ``/ai/models/search`` and merge with hardcoded ``model_facts``.
        """
        try:
            response = await self._http.get(f"{self._models_base}/models/search")
            response.raise_for_status()
            payload = response.json()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Cloudflare model discovery failed: %s", exc
            )
            return []

        raw_models = payload.get("result") or []
        results: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for model in raw_models:
            model_id = model.get("name") or model.get("id")
            if not isinstance(model_id, str):
                continue
            if not model_id.startswith("@cf/"):
                continue
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)

            facts = self._model_facts.get(model_id, {})
            entry: Dict[str, Any] = {
                "id": model_id,
                "display_name": (
                    model.get("description")
                    or facts.get("display_name")
                    or model_id
                )[:80],
                "type": "llm",
            }

            ctx_len = (
                facts.get("max_context_length")
                or _extract_property(model, "max_input_tokens")
            )
            if ctx_len:
                entry["max_context_length"] = ctx_len

            out_len = (
                facts.get("max_output_tokens")
                or _extract_property(model, "max_output_tokens")
            )
            if out_len:
                entry["max_output_tokens"] = out_len

            results.append(entry)

        # Ensure hardcoded models appear even if the live list did not
        # include them (e.g., deprecated-but-still-usable models).
        for model_id, facts in self._model_facts.items():
            if model_id in seen_ids:
                continue
            entry = {
                "id": model_id,
                "display_name": facts.get("display_name") or model_id,
                "type": "llm",
            }
            if facts.get("max_context_length"):
                entry["max_context_length"] = facts["max_context_length"]
            if facts.get("max_output_tokens"):
                entry["max_output_tokens"] = facts["max_output_tokens"]
            results.append(entry)

        return results

    # ------------------------------------------------------------------
    # ProviderAdapter: validate_connection
    # ------------------------------------------------------------------

    async def validate_connection(
        self, connection: Optional[ConnectionContext] = None
    ) -> Dict[str, Any]:
        """
        Ping ``/ai/models/search`` (cheap, no inference cost).

        Uses the ``per_page=1`` query parameter to minimize payload size.
        Returns ``{ok, message, latency_ms}`` — never raises.
        """
        started_at = time.time()
        try:
            response = await self._http.get(
                f"{self._models_base}/models/search",
                params={"per_page": 1},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {
                "ok": False,
                "message": str(exc),
                "latency_ms": round((time.time() - started_at) * 1000.0, 2),
            }

        latency_ms = round((time.time() - started_at) * 1000.0, 2)

        if response.status_code == 200:
            return {
                "ok": True,
                "message": "Cloudflare Workers AI reachable.",
                "latency_ms": latency_ms,
            }

        message = _short_body(response)
        if response.status_code == 401:
            message = f"Authentication failed: {message}"
        elif response.status_code == 403:
            message = f"Permission denied: {message}"
        elif response.status_code == 404:
            message = f"Account not found: {message}"

        return {
            "ok": False,
            "message": message,
            "latency_ms": latency_ms,
            "status_code": response.status_code,
        }

    # ------------------------------------------------------------------
    # ProviderAdapter: get_model_metadata
    # ------------------------------------------------------------------

    async def get_model_metadata(
        self, model_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return metadata for a specific model, or ``None``."""
        for model in await self.discover_models():
            if model["id"] == model_id:
                return model
        return None

    # ------------------------------------------------------------------
    # ProviderAdapter: close
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release both HTTP clients. Idempotent, never raises."""
        try:
            await self._client.close()
        except Exception as exc:
            logger.debug("OpenAI client close raised: %s", exc)

        try:
            await self._http.aclose()
        except Exception as exc:
            logger.debug("httpx client close raised: %s", exc)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _load_model_facts(self) -> Dict[str, Dict[str, Any]]:
        """
        Load ``model_facts`` from the package's ``manifest.json``.

        Same pattern as ``gigachat``'s adapter: the plugin descriptor
        provides ``package_dir``. Falls back to walking up from this
        module's location if the descriptor is unavailable (e.g., during
        local unit tests).
        """
        package_dir: Optional[str] = None

        raw_extra = self._connection.extra_params or {}
        plugin_info = raw_extra.get("_peldrun_plugin")
        if isinstance(plugin_info, dict):
            candidate = plugin_info.get("package_dir")
            if isinstance(candidate, str) and candidate:
                package_dir = candidate

        if not package_dir:
            # src/cloudflare_ai_peldrun/adapter.py -> package root
            package_dir = str(Path(__file__).parent.parent)

        try:
            manifest_path = Path(package_dir) / "manifest.json"
            if not manifest_path.is_file():
                logger.debug(
                    "No manifest.json found at %s; model_facts will be empty.",
                    manifest_path,
                )
                return {}

            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            facts = data.get("model_facts") or {}
            if not isinstance(facts, dict):
                return {}

            return {
                str(k): dict(v)
                for k, v in facts.items()
                if isinstance(v, dict)
            }
        except Exception as exc:
            logger.warning("Could not load model_facts: %s", exc)
            return {}

    def _build_chat_kwargs(
        self, compiled: CompiledRequest, *, stream: bool
    ) -> Dict[str, Any]:
        """Assemble kwargs for ``chat.completions.create``."""
        messages: List[Dict[str, Any]] = []
        if compiled.system_prompt:
            messages.append(
                {"role": "system", "content": compiled.system_prompt}
            )
        messages.extend(compiled.messages)

        kwargs: Dict[str, Any] = {
            "model": compiled.model,
            "messages": messages,
            "stream": stream,
        }

        # Generation parameters. Cloudflare Workers AI uses ``max_tokens``
        # (not ``max_completion_tokens``), so we intentionally send
        # ``max_tokens`` here even though it is deprecated upstream.
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "seed",
            "frequency_penalty",
            "presence_penalty",
        ):
            value = compiled.generation.get(key)
            if value is not None:
                kwargs[key] = value

        # Tools. Normalize the compiler's dict-form generic modes to the
        # string form OpenAI's SDK expects ("auto"/"none"/"required").
        if compiled.tools:
            kwargs["tools"] = compiled.tools
            if compiled.tool_choice:
                tc = compiled.tool_choice
                if isinstance(tc, dict) and tc.get("type") in (
                    "auto",
                    "none",
                    "required",
                ):
                    kwargs["tool_choice"] = tc["type"]
                else:
                    kwargs["tool_choice"] = tc

        # Structured output.
        if compiled.response_format:
            kwargs["response_format"] = compiled.response_format

        # Reasoning — Cloudflare supports ``reasoning_effort`` on some
        # models (QwQ, R1, Kimi). It must go through ``extra_body`` for
        # the OpenAI SDK.
        if compiled.reasoning:
            effort = compiled.reasoning.get("reasoning_effort")
            if effort:
                extra_body = dict(kwargs.get("extra_body") or {})
                extra_body["reasoning_effort"] = effort
                kwargs["extra_body"] = extra_body

        # Streaming usage: Cloudflare accepts ``stream_options`` on the
        # compatible endpoint for models that report usage.
        if stream:
            kwargs["stream_options"] = {"include_usage": True}

        # Provider-option escape hatch (e.g., rejectIfBusy).
        if compiled.provider_options:
            kwargs.update(compiled.provider_options)

        return kwargs

    def _convert_response(
        self, response: Any, model: str
    ) -> LLMResponse:
        """Convert an OpenAI ``ChatCompletion`` into ``LLMResponse``."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return LLMResponse(
                content="",
                model=getattr(response, "model", model) or model,
                raw=response,
                usage=_convert_usage(getattr(response, "usage", None)),
                finish_reason="stop",
            )

        first = choices[0]
        message = getattr(first, "message", None)

        # Content may be a plain string or a list of parts (rare).
        content = getattr(message, "content", None) if message else None
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )

        # Reasoning text extraction (some models expose it via model_extra).
        reasoning_text: Optional[str] = None
        if message is not None:
            extra = getattr(message, "model_extra", None) or {}
            if isinstance(extra, dict):
                for key in ("reasoning_content", "reasoning", "thinking"):
                    value = extra.get(key)
                    if isinstance(value, str) and value:
                        reasoning_text = value
                        break

        # Tool calls.
        tool_calls = _extract_tool_calls(message)

        finish_reason = getattr(first, "finish_reason", None)

        return LLMResponse(
            content=content or "",
            reasoning=reasoning_text,
            reasoning_content=reasoning_text,
            thought=reasoning_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=_convert_usage(getattr(response, "usage", None)),
            model=getattr(response, "model", model) or model,
            raw=response,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tool_calls(message: Any) -> List[ToolCall]:
    """Convert ``message.tool_calls`` to ``ToolCall`` instances."""
    if message is None:
        return []
    raw_calls = getattr(message, "tool_calls", None) or []
    result: List[ToolCall] = []
    for idx, tc in enumerate(raw_calls):
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", None) or ""
        raw_args = getattr(fn, "arguments", None)
        parsed_args: Any
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                parsed_args = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            parsed_args = raw_args
        else:
            parsed_args = {}
        result.append(
            ToolCall(
                id=getattr(tc, "id", None) or f"call_{idx}",
                name=name,
                arguments=parsed_args,
                type="function",
            )
        )
    return result


def _convert_usage(raw_usage: Any) -> Optional[TokenUsage]:
    """Convert OpenAI ``CompletionUsage`` into ``TokenUsage``."""
    if raw_usage is None:
        return None

    inp = getattr(raw_usage, "prompt_tokens", None)
    out = getattr(raw_usage, "completion_tokens", None)
    total = getattr(raw_usage, "total_tokens", None)

    cached: Optional[int] = None
    prompt_details = getattr(raw_usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached = getattr(prompt_details, "cached_tokens", None)

    reasoning: Optional[int] = None
    completion_details = getattr(raw_usage, "completion_tokens_details", None)
    if completion_details is not None:
        reasoning = getattr(completion_details, "reasoning_tokens", None)

    if inp is None and out is None and total is None:
        return None

    return TokenUsage(
        input_tokens=int(inp) if inp is not None else None,
        output_tokens=int(out) if out is not None else None,
        total_tokens=int(total) if total is not None else (
            int(inp or 0) + int(out or 0)
        ),
        cached_input_tokens=int(cached) if cached is not None else None,
        reasoning_output_tokens=int(reasoning) if reasoning is not None else None,
        source=UsageSource.PROVIDER,
        estimated=False,
    )


def _extract_property(model: Dict[str, Any], property_id: str) -> Optional[int]:
    """
    Extract a numeric property from the Cloudflare model search payload.

    Cloudflare's response shape:
        "properties": [{"property_id": "max_input_tokens", "value": 32000}, ...]
    """
    props = model.get("properties") or []
    if not isinstance(props, list):
        return None
    for entry in props:
        if not isinstance(entry, dict):
            continue
        if entry.get("property_id") != property_id:
            continue
        value = entry.get("value")
        if value is None:
            continue
        try:
            return int(value)
        except (ValueError, TypeError):
            continue
    return None


def _short_body(response: httpx.Response, limit: int = 200) -> str:
    """Return a truncated body for error messages."""
    try:
        text = response.text
    except Exception:
        return f"HTTP {response.status_code}"
    text = text.strip()
    if not text:
        return f"HTTP {response.status_code}"
    return text[:limit]


def _map_openai_exception(
    exc: BaseException, *, provider: str, model: str
) -> ProviderError:
    """Translate an OpenAI SDK exception into a canonical ProviderError."""
    if isinstance(exc, openai.AuthenticationError):
        return ProviderAuthError(
            str(exc), provider=provider, model=model, status_code=401
        )
    if isinstance(exc, openai.PermissionDeniedError):
        return ProviderAuthError(
            str(exc), provider=provider, model=model, status_code=403
        )
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError(
            str(exc), provider=provider, model=model, status_code=429
        )
    if isinstance(exc, openai.APITimeoutError):
        return ProviderTimeoutError(
            str(exc), provider=provider, model=model, phase="request"
        )
    if isinstance(exc, openai.APIConnectionError):
        return ProviderConnectionError(
            str(exc), provider=provider, model=model
        )
    if isinstance(exc, openai.BadRequestError):
        return ProviderInvalidRequestError(
            str(exc), provider=provider, model=model, status_code=400
        )
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", 500) or 500
        if status >= 500:
            return ProviderUnavailableError(
                str(exc), provider=provider, model=model, status_code=status
            )
        return ProviderInvalidRequestError(
            str(exc), provider=provider, model=model, status_code=status
        )
    if isinstance(exc, openai.APIError):
        return ProviderUnavailableError(
            str(exc), provider=provider, model=model
        )

    return ProviderError(
        f"{type(exc).__name__}: {exc}", provider=provider, model=model
    )


def _error_event(err: ProviderError, *, sequence: int) -> LLMStreamEvent:
    """Wrap a ``ProviderError`` in a ``PROVIDER_ERROR`` stream event."""
    return LLMStreamEvent(
        type=StreamEventType.PROVIDER_ERROR,
        error=err.to_dict(),
        sequence=sequence,
    )


__all__ = ["CloudflareAIAdapter"]