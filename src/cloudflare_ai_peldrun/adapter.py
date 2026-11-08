# backend/peldrun/llm/plugin_host/permissions.py
"""
Runtime permission enforcement for Provider Package workers.

This module is the enforcement surface that the worker uses to
restrict what an adapter can actually do at runtime. It is
deliberately separate from the DECLARATION surface in
``omweb.provider_packages.permissions`` (which parses and validates
the manifest) because:

    * The declaration lives in the admin layer (OMWeb) and evolves
      with the manifest schema.
    * The enforcement lives in the runtime layer (peldrun.llm) and
      must remain stable across schema changes.
    * The two must not import each other — the architectural boundary
      between OMWeb and peldrun.llm is strict (see ``contracts.py``).

The runtime enforcer receives the flat permission string list from
the IPC ``LOAD_ADAPTER`` payload (which the supervisor derived from
the manifest) and enforces three guarantees:

    1. **Network allowlist.** The adapter may only reach the
       connection's configured ``base_url`` — scheme, host, and port
       must match exactly. This prevents SSRF (an adapter cannot read
       AWS instance metadata at ``169.254.169.254`` or the local
       PELDRUN admin API at ``127.0.0.1:8080``).

    2. **Secret isolation.** The adapter receives ITS OWN
       connection's ``api_key`` and nothing else. There is no
       mechanism to request another connection's key, because that
       data was never passed to the worker in the first place. This
       module adds defense-in-depth: ``get_api_key()`` returns
       ``None`` if the manifest did not declare
       ``secrets.read:connection``, even though the key is technically
       present in the connection context.

    3. **Forbidden permissions.** ``shell.execute``,
       ``workspace.read``, and ``workspace.write`` are rejected at
       manifest parse time (see ``omweb.provider_packages.schema``),
       but this module re-asserts the rejection at runtime. A
       corrupted IPC payload cannot smuggle them through.

Threat model and limitations
----------------------------

This module is a best-effort runtime guard, NOT a security sandbox.
Python code that obtains an interpreter can bypass any in-process
check via reflection or by spawning its own socket. The real
isolation guarantees come from:

    * Process separation (worker cannot touch the main process's
      memory or file descriptors).
    * Environment isolation (worker runs with the package's own venv).
    * Least-privilege data passing (worker never receives other
      connections' secrets).

OS-level sandboxing (seccomp, AppContainer, entitlements) is a
future enhancement. The enforcer API is designed to work with it —
``AllowlistAsyncTransport`` is exactly the kind of hook a real
firewall would implement, so swapping the implementation later is a
drop-in change.

The enforcer is designed to be TRUE positives only: every call to
``assert_*`` that succeeds is genuinely allowed by the manifest. A
call that fails is genuinely not allowed. There is no "unknown" or
"probable" state.

Design invariants
-----------------

1. **No silent failures.** Every assertion either returns silently
   (allowed) or raises ``PluginPermissionDeniedError`` with a
   structured message. There is no boolean return that the caller
   could accidentally ignore.

2. **Fail closed.** When the enforcer cannot make a decision (e.g.
   the manifest declared no network permission but a URL check is
   requested), it denies. When it can make a decision but the
   decision is confusing (e.g. ``base_url`` is empty string), it
   denies. Silence is only returned when the operation is
   unambiguously allowed.

3. **Local host equivalence.** ``localhost``, ``127.0.0.1``, and
   ``::1`` are treated as the same host when the base URL is one of
   them. This matters because LM Studio and Ollama bind to different
   local addresses depending on the platform, and adapters should not
   care.

4. **Redirect targets are checked.** The allowlist transport
   intercepts EVERY request, including redirect-following requests
   that httpx creates internally. A ``302 Location: https://evil.com``
   from a compromised provider endpoint is caught at the second hop.

5. **Secrets are redacted in every diagnostic path.** The
   ``redacted_dump()`` and ``redacted_secrets()`` methods are the
   ONLY way to serialize the enforcer's state for logging. They
   replace the actual ``api_key`` value with ``"<redacted>"``. No
   method ever returns the raw key except ``get_api_key()``, which
   requires the permission to be declared.

6. **No httpx import at module level.** The transport wrapper needs
   ``httpx``, but importing it eagerly would prevent this module from
   being used in contexts where httpx is not installed (e.g. an
   adapter that talks raw sockets). The import is lazy inside the
   ``AllowlistAsyncTransport`` class body.

See also
--------

    * ``backend/peldrun/llm/plugin_host/worker.py`` — installs the
      enforcer before loading the adapter
    * ``backend/peldrun/llm/errors.py`` — canonical error taxonomy
    * ``backend/omweb/provider_packages/permissions.py`` — declaration
      surface (must not be imported from here)
    * ``backend/omweb/provider_packages/schema.py`` —
      ``PermissionKind`` enum (mirrored as plain strings here)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from peldrun.llm.errors import ProviderError

logger = logging.getLogger("peldrun.llm.plugin_host.permissions")


# ---------------------------------------------------------------------------
# Permission string constants
# ---------------------------------------------------------------------------
#
# These are the string values that appear in ``LoadAdapterPayload.permissions``
# over IPC. They mirror ``omweb.provider_packages.schema.PermissionKind``
# exactly, but are declared here as plain strings to preserve the
# architecture boundary (peldrun.llm must not import from omweb).
#
# A mismatch between these constants and the schema enum is caught by
# the test suite (``tests/llm/plugin_host/test_permission_constants.py``).
# Keep them in sync when adding new permissions.

PERM_NETWORK_CONNECT: str = "network.connect"
PERM_SECRETS_READ_CONNECTION: str = "secrets.read:connection"
PERM_MODELS_DISCOVER: str = "models.discover"
PERM_LLM_GENERATE: str = "llm.generate"
PERM_LLM_STREAM: str = "llm.stream"
PERM_USAGE_READ: str = "usage.read"

PERM_WORKSPACE_READ: str = "workspace.read"
PERM_WORKSPACE_WRITE: str = "workspace.write"
PERM_SHELL_EXECUTE: str = "shell.execute"


#: Permissions that can NEVER be granted to a Provider Package, no
#: matter what the manifest says. Mirrors
#: ``omweb.provider_packages.schema.FORBIDDEN_PERMISSIONS``.
FORBIDDEN_PERMISSIONS: FrozenSet[str] = frozenset(
    {
        PERM_WORKSPACE_READ,
        PERM_WORKSPACE_WRITE,
        PERM_SHELL_EXECUTE,
    }
)


#: Every permission string this build understands. Any string that
#: appears in a worker's IPC payload but is NOT in this set triggers a
#: load-time rejection — the runtime refuses to load a package whose
#: manifest declares a permission the runtime does not recognize.
KNOWN_PERMISSIONS: FrozenSet[str] = frozenset(
    {
        PERM_NETWORK_CONNECT,
        PERM_SECRETS_READ_CONNECTION,
        PERM_MODELS_DISCOVER,
        PERM_LLM_GENERATE,
        PERM_LLM_STREAM,
        PERM_USAGE_READ,
    }
) | FORBIDDEN_PERMISSIONS


#: Hostnames that are considered equivalent to each other when the
#: base URL is one of them. ``localhost``, IPv4 loopback, and IPv6
#: loopback all resolve to "this machine" and adapters should not
#: need to know which literal form the provider uses.
_LOCAL_HOSTS: FrozenSet[str] = frozenset(
    {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
)


#: Metadata service IPs that are ALWAYS denied, regardless of the base
#: URL. These are classic SSRF targets: a compromised adapter could
#: query the cloud provider's instance metadata endpoint to obtain
#: IAM credentials. Even if a user misconfigures the base URL to one
#: of these, we refuse — the risk is too high.
_DENYLISTED_HOSTS: FrozenSet[str] = frozenset(
    {
        "169.254.169.254",        # AWS / GCP / Azure instance metadata
        "metadata.google.internal",
        "169.254.170.2",          # ECS task metadata
        "100.100.100.200",        # Alibaba Cloud metadata
    }
)


#: Provider-specific additional hosts that the adapter is allowed to
#: reach BEYOND the connection's base_url. This exists because some
#: providers use a DIFFERENT host:port for authentication than for
#: API calls — a single base_url allowlist cannot cover them.
#:
#: GigaChat is the canonical example:
#:   * OAuth token exchange:
#:       https://ngw.devices.sberbank.ru:9443/api/v2/oauth
#:   * API endpoint (base_url):
#:       https://gigachat.devices.sberbank.ru/api/v1
#:
#: Both are required for the adapter to function. Without allowing
#: the OAuth host, the SDK raises URLNotAllowedError before it can
#: even retrieve an access token.
#:
#: In a future PELDRUN release, this list will move into the package
#: manifest (as an `additional_allowed_hosts` field), so each package
#: declares its own needs rather than relying on a hardcoded table.
#: For now, this is the minimal fix that keeps existing packages
#: working without a schema change.
_PROVIDER_EXTRA_HOSTS: Dict[str, Tuple[Tuple[str, str, int], ...]] = {
    "gigachat": (
        # OAuth token exchange server — different host:port from the
        # API endpoint, and GigaChat SDK creates its own HTTP client
        # for it that bypasses the connection's base_url.
        ("https", "ngw.devices.sberbank.ru", 9443),
        # Official GigaChat API endpoint (since July 2026). The legacy
        # endpoint (gigachat.devices.sberbank.ru) issues an HTTP
        # redirect to this host, and the GigaChat SDK follows
        # redirects internally. Without this entry, any request
        # against the legacy endpoint fails at the redirect hop with
        # "host 'api.giga.chat' is not in the connection's allowlist".
        ("https", "api.giga.chat", 443),
    ),
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PluginPermissionDeniedError(ProviderError):
    """
    A permission the adapter attempted to use was not granted.

    Distinct from ``ProviderPermissionError`` (which represents a 403
    from the provider): this one is raised by PELDRUN's own runtime
    enforcer BEFORE any network call leaves the worker.

    The supervisor rehydrates this into the same exception class in
    the main process, so callers see a uniform error regardless of
    where the failure occurred.

    ``retryable`` is ``False`` — retrying with the same manifest will
    fail identically.
    """

    code = "plugin_permission_denied"
    retryable = False


class URLNotAllowedError(PluginPermissionDeniedError):
    """
    The adapter tried to reach a URL outside the connection's
    configured ``base_url``.

    This is a subclass of ``PluginPermissionDeniedError`` so callers
    that catch the broader class still work, but the distinct code
    lets the UI display a specific "SSRF blocked" message with the
    attempted URL (redacted).

    ``retryable`` is ``False`` — the URL is either in the allowlist
    or it is not; retrying cannot change that.
    """

    code = "plugin_url_not_allowed"
    retryable = False


# ---------------------------------------------------------------------------
# URL normalization helpers
# ---------------------------------------------------------------------------


def _normalize_url_parts(url: str) -> Tuple[str, str, int]:
    """
    Parse a URL into ``(scheme, host, port)`` with defaults filled in.

    Rules
    -----
    * Scheme is lowercased.
    * Host is lowercased and stripped of any trailing dot (some DNS
      conventions write ``api.openai.com.`` with a trailing dot; this
      is equivalent to the dotless form).
    * Port defaults to ``443`` for ``https`` and ``80`` for ``http``.
    * Empty host raises ``ValueError`` — an empty host is not a URL.

    The function deliberately does NOT include the path, query, or
    fragment — those are checked separately (or not at all: the
    allowlist operates at host:port granularity).

    Raises
    ------
    ValueError
        When the URL is not parseable or has no host component.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"Could not parse URL {url!r}: {exc}") from exc

    scheme = (parsed.scheme or "").lower().strip()
    host = (parsed.hostname or "").lower().strip()
    if host.endswith("."):
        host = host[:-1]

    if not scheme:
        raise ValueError(f"URL {url!r} has no scheme.")
    if scheme not in ("http", "https"):
        raise ValueError(
            f"URL {url!r} has unsupported scheme {scheme!r}. "
            f"Only http and https are allowed."
        )
    if not host:
        raise ValueError(f"URL {url!r} has no host.")

    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80

    return scheme, host, port


def _hosts_are_equivalent(a: str, b: str) -> bool:
    """
    Return True iff ``a`` and ``b`` refer to the same host for the
    purposes of the allowlist.

    Two hosts are equivalent when:
        * They are exactly equal (after normalization).
        * Both are members of ``_LOCAL_HOSTS`` — ``localhost``,
          ``127.0.0.1``, ``::1``, and ``0.0.0.0`` all mean "this
          machine" and adapters should not care which literal the
          provider uses.

    The local-equivalence rule matters in practice: LM Studio often
    binds to ``127.0.0.1`` on Windows and ``localhost`` on macOS, and
    Ollama does the same. Without this rule, a connection whose
    ``base_url`` is ``http://localhost:11434`` would reject an
    adapter that hits ``http://127.0.0.1:11434/api/chat``.
    """
    if a == b:
        return True
    if a in _LOCAL_HOSTS and b in _LOCAL_HOSTS:
        return True
    return False


def _redact_url(url: str) -> str:
    """
    Return a version of ``url`` safe to include in error messages.

    Redacts userinfo (``user:password@host``) and query parameters
    that commonly carry secrets (``api_key``, ``token``, ``access_token``,
    ``key``). The host and path are preserved because they are useful
    for debugging: knowing that the adapter tried to reach
    ``https://evil.com/v1/chat`` is more diagnostic than
    ``<redacted>``.

    Never raises — a malformed URL is returned as a truncated
    representation.
    """
    try:
        parsed = urlparse(url)
        if not parsed.hostname:
            return f"<unparseable-url: {url[:100]!r}>"

        # Rebuild without userinfo and with redacted query params.
        scheme = parsed.scheme or "?"
        host = parsed.hostname
        port_part = f":{parsed.port}" if parsed.port else ""
        path = parsed.path or ""

        if parsed.query:
            # Redact common secret-bearing parameter names.
            redacted_params: List[str] = []
            secret_names = {
                "api_key",
                "key",
                "token",
                "access_token",
                "auth",
                "password",
                "secret",
            }
            for pair in parsed.query.split("&"):
                if "=" in pair:
                    name, _, _value = pair.partition("=")
                    if name.lower() in secret_names:
                        redacted_params.append(f"{name}=<redacted>")
                    else:
                        redacted_params.append(pair)
                else:
                    redacted_params.append(pair)
            query_part = "?" + "&".join(redacted_params)
        else:
            query_part = ""

        return f"{scheme}://{host}{port_part}{path}{query_part}"
    except Exception:
        return f"<unparseable-url: {url[:100]!r}>"


# ---------------------------------------------------------------------------
# Runtime enforcer
# ---------------------------------------------------------------------------


class RuntimePermissionEnforcer:
    """
    The worker-side permission enforcer.

    Constructed ONCE per worker, immediately after ``LOAD_ADAPTER``
    arrives over IPC. Stores:

        * The manifest's declared permissions (already validated by
          the admin layer; re-validated here as defense-in-depth).
        * The connection's ``base_url`` — the ONLY host the adapter
          may reach over the network.
        * The connection's ``api_key`` — the ONLY secret the adapter
          may read (and only when ``secrets.read:connection`` is
          granted).
        * The connection's ``connection_id`` — used in error
          messages to identify which connection was violated.

    The class is designed to be passed to:
        * ``worker._handle_load_adapter()`` — validates the payload.
        * Every adapter method the worker dispatches — via
          ``require_*`` calls.
        * ``install_httpx_guard()`` — wraps the adapter's httpx client.
        * Any custom adapter code that wants to call
          ``assert_url_allowed()`` explicitly before a non-httpx
          network call.

    Thread safety
    -------------
    All fields are read-only after construction. Reads are safe from
    any number of concurrent tasks without locking. There is no
    mutable state.
    """

    def __init__(
        self,
        *,
        provider_id: str,
        connection_id: str,
        declared_permissions: Iterable[str],
        connection_base_url: Optional[str],
        connection_api_key: Optional[str],
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        provider_id
            The package's canonical ID (from the manifest).
        connection_id
            The connection's ID, used only in error messages so a
            denied permission can be attributed to the correct
            connection.
        declared_permissions
            The flat string set from the manifest. Any string that
            appears here but is NOT in ``KNOWN_PERMISSIONS`` triggers
            an immediate construction failure — a runtime never
            accepts a permission it does not understand.
        connection_base_url
            The connection's ``base_url``. This is the ONLY host:port
            the adapter may reach over the network. When ``None`` or
            empty, the adapter may not perform ANY network I/O,
            regardless of what the manifest declares. This is the
            fail-closed rule: no endpoint means no network.
        connection_api_key
            The connection's API key. Present in the connection
            context regardless of the manifest's permissions — but
            ``get_api_key()`` returns it ONLY when
            ``secrets.read:connection`` is granted. This lets the
            enforcer distinguish "the adapter does not need the key"
            from "the adapter needs it but is not allowed to read it."
        extra_headers
            Additional headers from the connection context. These are
            included in ``redacted_dump()`` for diagnostics but never
            used for authorization. Only present so the operator can
            see what headers the adapter would forward.

        Raises
        ------
        PluginPermissionDeniedError
            When ``declared_permissions`` contains a value that is
            forbidden (``shell.execute``, ``workspace.*``) or unknown.
        """
        self._provider_id = provider_id
        self._connection_id = connection_id

        # Validate the declared permission set. Any forbidden value is
        # a hard error — the manifest validation should have caught
        # this already, but we do not trust the IPC boundary.
        declared_set: FrozenSet[str] = frozenset(declared_permissions or ())
        forbidden_encountered = declared_set & FORBIDDEN_PERMISSIONS
        if forbidden_encountered:
            raise PluginPermissionDeniedError(
                f"Provider package '{provider_id}' declares forbidden "
                f"permissions {sorted(forbidden_encountered)}. These "
                f"are never granted to Provider Packages.",
                details={
                    "provider_id": provider_id,
                    "forbidden": sorted(forbidden_encountered),
                },
            )

        unknown_encountered = declared_set - KNOWN_PERMISSIONS
        if unknown_encountered:
            raise PluginPermissionDeniedError(
                f"Provider package '{provider_id}' declares "
                f"unrecognized permissions "
                f"{sorted(unknown_encountered)}. The runtime does not "
                f"know what those permissions mean and cannot enforce "
                f"them, so the package cannot be loaded.",
                details={
                    "provider_id": provider_id,
                    "unknown": sorted(unknown_encountered),
                    "known": sorted(KNOWN_PERMISSIONS),
                },
            )

        self._permissions = declared_set

        # Parse the base URL. A malformed base URL means the adapter
        # cannot reach anything — we do not silently disable the
        # network, we make every URL check fail loudly.
        self._allowed_base_url: Optional[str] = None
        self._allowed_parts: Optional[Tuple[str, str, int]] = None

        if connection_base_url:
            try:
                parts = _normalize_url_parts(connection_base_url)
                self._allowed_parts = parts
                self._allowed_base_url = (
                    f"{parts[0]}://{parts[1]}:{parts[2]}"
                )
            except ValueError as exc:
                # Log the failure but do not raise. The adapter will
                # still load (some adapters do no network I/O); every
                # subsequent network attempt will fail with a clear
                # "no endpoint" error.
                logger.error(
                    "Connection '%s' has an unparseable base_url "
                    "(%r): %s. All network calls will be denied.",
                    connection_id,
                    connection_base_url,
                    exc,
                )
                self._allowed_parts = None
                self._allowed_base_url = None

        # Provider-specific extra allowed (scheme, host, port) triples.
        # See the _PROVIDER_EXTRA_HOSTS table above for rationale.
        # Empty tuple when the provider has no extras declared.
        self._extra_allowed_parts: Tuple[Tuple[str, str, int], ...] = (
            _PROVIDER_EXTRA_HOSTS.get(provider_id, ())
        )

        self._api_key = connection_api_key
        self._extra_headers: Dict[str, str] = dict(extra_headers or {})

        logger.debug(
            "RuntimePermissionEnforcer ready: provider=%r "
            "connection=%r permissions=%s base_url=%s extra_hosts=%s",
            provider_id,
            connection_id,
            sorted(self._permissions),
            self._allowed_base_url or "<none>",
            [
                f"{s}://{h}:{p}"
                for s, h, p in self._extra_allowed_parts
            ]
            or "<none>",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def connection_id(self) -> str:
        return self._connection_id

    @property
    def permissions(self) -> FrozenSet[str]:
        """The declared permission set (already validated)."""
        return self._permissions

    @property
    def allowed_base_url(self) -> Optional[str]:
        """The normalized ``scheme://host:port`` the adapter may reach."""
        return self._allowed_base_url

    # ------------------------------------------------------------------
    # Permission checks
    # ------------------------------------------------------------------

    def has(self, permission: str) -> bool:
        """Return True iff ``permission`` is granted."""
        return permission in self._permissions

    def require(self, permission: str) -> None:
        """
        Raise ``PluginPermissionDeniedError`` if ``permission`` is not
        granted.

        Use this at the top of every adapter method the worker
        dispatches — even when the permission seems implied by another
        one. For example, ``generate`` requires BOTH ``llm.generate``
        AND ``network.connect`` (a generate call is meaningless
        without a network). The worker calls both checks explicitly
        so that a manifest which declares one but not the other is
        rejected clearly.
        """
        if permission not in self._permissions:
            raise PluginPermissionDeniedError(
                f"Provider package '{self._provider_id}' (connection "
                f"'{self._connection_id}') attempted to use "
                f"permission '{permission}', which was not declared in "
                f"its manifest.",
                provider=self._provider_id,
                details={
                    "requested": permission,
                    "granted": sorted(self._permissions),
                    "connection_id": self._connection_id,
                },
            )

    def require_any(self, *permissions: str) -> str:
        """
        Raise ``PluginPermissionDeniedError`` if NONE of the given
        permissions are granted.

        Returns the first granted permission from the list (useful for
        logging which specific path was taken). If the list is empty,
        raises ``ValueError`` — an empty "any" is a programming bug.
        """
        if not permissions:
            raise ValueError(
                "require_any() requires at least one permission name."
            )
        for perm in permissions:
            if perm in self._permissions:
                return perm
        raise PluginPermissionDeniedError(
            f"Provider package '{self._provider_id}' (connection "
            f"'{self._connection_id}') attempted to use one of "
            f"{sorted(permissions)}, but none of them were declared in "
            f"its manifest.",
            provider=self._provider_id,
            details={
                "requested_any": sorted(permissions),
                "granted": sorted(self._permissions),
                "connection_id": self._connection_id,
            },
        )

    # ------------------------------------------------------------------
    # Network allowlist
    # ------------------------------------------------------------------

    def is_url_allowed(self, url: str) -> bool:
        """
        Return True iff ``url`` is reachable under the current
        permissions and connection configuration.

        A URL is allowed iff ALL of the following hold:

            1. ``network.connect`` is granted.
            2. The connection has a valid ``base_url``.
            3. The URL parses successfully.
            4. The URL's host is NOT in the hardcoded metadata
               denylist.
            5. The URL's ``(scheme, host, port)`` matches EITHER the
               connection's ``base_url`` OR one of the provider's
               declared extra hosts (see ``_PROVIDER_EXTRA_HOSTS``),
               with the local-host equivalence rule applied to the
               host in both cases.

        Returns ``False`` (never raises) when any of the above fails.
        Use ``assert_url_allowed()`` for the raising version.

        The method is intentionally total: it returns a boolean for
        ANY input, including empty strings, malformed URLs, and
        non-http schemes. The caller never has to special-case
        anything.
        """
        if PERM_NETWORK_CONNECT not in self._permissions:
            return False
        if self._allowed_parts is None:
            return False

        try:
            scheme, host, port = _normalize_url_parts(url)
        except ValueError:
            return False

        if host in _DENYLISTED_HOSTS:
            return False

        # Primary allowlist — the connection's base_url.
        allowed_scheme, allowed_host, allowed_port = self._allowed_parts
        if (
            scheme == allowed_scheme
            and port == allowed_port
            and _hosts_are_equivalent(host, allowed_host)
        ):
            return True

        # Provider-specific extra hosts (e.g. GigaChat's OAuth server).
        for extra_scheme, extra_host, extra_port in self._extra_allowed_parts:
            if (
                scheme == extra_scheme
                and port == extra_port
                and _hosts_are_equivalent(host, extra_host)
            ):
                return True

        return False

    def assert_url_allowed(self, url: str) -> None:
        """
        Raise ``URLNotAllowedError`` if ``url`` is not allowed.

        The error message names the exact reason the URL was
        rejected — missing permission, missing base_url, metadata
        IP, different host, different port, different scheme, or
        malformed URL — so the operator can diagnose quickly.

        The attempted URL is included in REDACTED form (userinfo and
        secret query params removed). Logging the raw URL would leak
        an adapter's accidentally-hardcoded API key.
        """
        if PERM_NETWORK_CONNECT not in self._permissions:
            raise URLNotAllowedError(
                f"Provider package '{self._provider_id}' (connection "
                f"'{self._connection_id}') attempted a network call "
                f"to '{_redact_url(url)}' but did not declare the "
                f"'network.connect' permission.",
                provider=self._provider_id,
                details={
                    "attempted_url": _redact_url(url),
                    "reason": "no_network_permission",
                    "connection_id": self._connection_id,
                },
            )

        if self._allowed_parts is None:
            raise URLNotAllowedError(
                f"Provider package '{self._provider_id}' (connection "
                f"'{self._connection_id}') attempted a network call "
                f"to '{_redact_url(url)}' but the connection has no "
                f"valid base_url configured. All network I/O is "
                f"disabled for this connection.",
                provider=self._provider_id,
                details={
                    "attempted_url": _redact_url(url),
                    "reason": "no_base_url",
                    "connection_id": self._connection_id,
                },
            )

        try:
            scheme, host, port = _normalize_url_parts(url)
        except ValueError as exc:
            raise URLNotAllowedError(
                f"Provider package '{self._provider_id}' (connection "
                f"'{self._connection_id}') attempted a network call "
                f"to an invalid URL ({exc}).",
                provider=self._provider_id,
                details={
                    "attempted_url": _redact_url(url),
                    "reason": "invalid_url",
                    "connection_id": self._connection_id,
                },
            ) from exc

        if host in _DENYLISTED_HOSTS:
            raise URLNotAllowedError(
                f"Provider package '{self._provider_id}' (connection "
                f"'{self._connection_id}') attempted to reach the "
                f"blocked metadata host '{host}'. This is never "
                f"allowed, regardless of the base_url configuration.",
                provider=self._provider_id,
                details={
                    "attempted_url": _redact_url(url),
                    "reason": "denylisted_host",
                    "host": host,
                    "connection_id": self._connection_id,
                },
            )

        allowed_scheme, allowed_host, allowed_port = self._allowed_parts

        # FIX: Primary allowlist check (was missing).
        # Without this, no provider can reach its own base_url
        # unless it also appears in _PROVIDER_EXTRA_HOSTS.
        if (
            scheme == allowed_scheme
            and port == allowed_port
            and _hosts_are_equivalent(host, allowed_host)
        ):
            return  # Allowed by primary allowlist.

        # Try the provider's extra allowed hosts BEFORE raising.
        # This is what lets GigaChat reach its OAuth server (a
        # different host:port from the API endpoint).
        for extra_scheme, extra_host, extra_port in self._extra_allowed_parts:
            if (
                scheme == extra_scheme
                and port == extra_port
                and _hosts_are_equivalent(host, extra_host)
            ):
                # Match — no exception.
                return

        # Neither primary nor extras matched. Pick the most specific
        # mismatch reason to surface to the user. Extras are included
        # in the error message so the operator sees what hosts WERE
        # allowed (both the base_url and any provider extras).
        allowed_summary = (
            [f"{allowed_scheme}://{allowed_host}:{allowed_port}"]
            + [
                f"{s}://{h}:{p}"
                for s, h, p in self._extra_allowed_parts
            ]
        )

        if scheme != allowed_scheme:
            raise URLNotAllowedError(
                f"Provider package '{self._provider_id}' (connection "
                f"'{self._connection_id}') attempted a {scheme!r} "
                f"request, but the connection's allowlist uses "
                f"{allowed_scheme!r}. Scheme must match.",
                provider=self._provider_id,
                details={
                    "attempted_url": _redact_url(url),
                    "attempted_scheme": scheme,
                    "allowed_scheme": allowed_scheme,
                    "allowed_hosts": allowed_summary,
                    "reason": "scheme_mismatch",
                    "connection_id": self._connection_id,
                },
            )

        if port != allowed_port:
            raise URLNotAllowedError(
                f"Provider package '{self._provider_id}' (connection "
                f"'{self._connection_id}') attempted to reach port "
                f"{port}, which is not in the connection's allowlist "
                f"(allowed: {allowed_summary}).",
                provider=self._provider_id,
                details={
                    "attempted_url": _redact_url(url),
                    "attempted_port": port,
                    "allowed_port": allowed_port,
                    "allowed_hosts": allowed_summary,
                    "reason": "port_mismatch",
                    "connection_id": self._connection_id,
                },
            )

        raise URLNotAllowedError(
            f"Provider package '{self._provider_id}' (connection "
            f"'{self._connection_id}') attempted to reach host "
            f"'{host}', which is not in the connection's allowlist "
            f"(allowed: {allowed_summary}).",
            provider=self._provider_id,
            details={
                "attempted_url": _redact_url(url),
                "attempted_host": host,
                "allowed_host": allowed_host,
                "allowed_hosts": allowed_summary,
                "reason": "host_mismatch",
                "connection_id": self._connection_id,
            },
        )

    # ------------------------------------------------------------------
    # Secret access
    # ------------------------------------------------------------------

    def get_api_key(self) -> Optional[str]:
        """
        Return the connection's API key IF
        ``secrets.read:connection`` is granted, else ``None``.

        This is the ONLY method that returns the raw secret value.
        Every other diagnostic method (``redacted_dump``,
        ``redacted_secrets``, ``__repr__``) redacts it.

        Returning ``None`` when the permission is missing is
        deliberate: an adapter that tries to use the key without the
        permission gets a silent ``None`` (and, if it then tries to
        send an ``Authorization: Bearer None`` header, the provider
        will return 401). The alternative — raising — would make the
        enforcer's behavior depend on the caller's error handling,
        which is exactly the kind of implicit state we want to avoid.
        Adapters SHOULD call ``require(PERM_SECRETS_READ_CONNECTION)``
        explicitly before the first read.
        """
        if PERM_SECRETS_READ_CONNECTION not in self._permissions:
            return None
        return self._api_key

    def secrets_available(self) -> bool:
        """
        Return True iff the adapter is allowed to read its own
        connection's API key.

        Used by adapters that want to skip an entire authentication
        code path when the key is not needed (e.g. local providers
        that accept unauthenticated requests).
        """
        return (
            PERM_SECRETS_READ_CONNECTION in self._permissions
            and self._api_key is not None
        )

    # ------------------------------------------------------------------
    # Redacted serialization
    # ------------------------------------------------------------------

    def redacted_dump(self) -> Dict[str, Any]:
        """
        Return a dict safe to log or transmit over diagnostics.

        The ``api_key`` is replaced with ``"<redacted>"`` if present,
        and the ``extra_headers`` dict has any recognized auth-style
        header names replaced. Every other field is included verbatim
        because they are not secrets.

        This is the ONLY way external code should inspect the
        enforcer's state. ``__dict__`` access is not part of the
        contract and its output is not sanitized.
        """
        redacted_headers = dict(self._extra_headers)
        for header_name in list(redacted_headers.keys()):
            if header_name.lower() in (
                "authorization",
                "x-api-key",
                "x-goog-api-key",
                "api-key",
                "proxy-authorization",
            ):
                redacted_headers[header_name] = "<redacted>"

        return {
            "provider_id": self._provider_id,
            "connection_id": self._connection_id,
            "permissions": sorted(self._permissions),
            "allowed_base_url": self._allowed_base_url,
            "api_key_present": self._api_key is not None,
            "api_key": "<redacted>" if self._api_key else None,
            "extra_headers": redacted_headers,
        }

    def __repr__(self) -> str:
        return (
            f"RuntimePermissionEnforcer(provider_id={self._provider_id!r}, "
            f"connection_id={self._connection_id!r}, "
            f"permissions={sorted(self._permissions)!r}, "
            f"allowed_base_url={self._allowed_base_url!r})"
        )


# ---------------------------------------------------------------------------
# httpx transport wrapper
# ---------------------------------------------------------------------------


def _import_httpx():
    """
    Lazy import of httpx.

    Imported inside the function (rather than at module scope) so
    that adapter code which does not use httpx can still import this
    module without httpx being installed. The plugin_host's own
    dependency chain includes httpx (via peldrun's dependencies), but
    being defensive costs nothing.
    """
    import httpx  # noqa: PLC0415 — intentional lazy import

    return httpx


# ---------------------------------------------------------------------------
# AllowlistAsyncTransport — dynamic class builder
# ---------------------------------------------------------------------------
#
# httpx.AsyncBaseTransport is a class with a non-default deallocator.
# Python forbids changing __bases__ at runtime when the new base's
# deallocator differs from the current one:
#
#   TypeError: __bases__ assignment: 'AsyncBaseTransport' deallocator
#              differs from 'object'
#
# That error was crashing every plugin worker on the first LOAD_ADAPTER.
#
# The correct fix is to build the entire subclass inside a factory
# function that runs on first use. The resulting class inherits from
# httpx.AsyncBaseTransport properly, so httpx.AsyncClient's
# `isinstance(transport, AsyncBaseTransport)` check passes.

_ALLOWLIST_TRANSPORT_CLASS: Optional[type] = None


def _build_allowlist_transport_class() -> type:
    """
    Build the real ``AllowlistAsyncTransport`` class.

    Deferred until first use so this module stays importable without
    httpx. The built class is cached (see
    ``_get_allowlist_transport_class``), so subsequent calls are O(1).
    """
    httpx = _import_httpx()

    class _AllowlistAsyncTransport(httpx.AsyncBaseTransport):
        """
        httpx transport that enforces the runtime permission enforcer's
        URL allowlist on EVERY request.

        Wraps an inner transport (typically ``httpx.AsyncHTTPTransport()``)
        and delegates every allowed request to it unchanged. This
        preserves TLS verification, proxy support, connection pooling,
        and HTTP/2 negotiation — only the URL check is new.

        Redirect targets are checked too: httpx's redirect loop
        creates a new request for each ``Location`` header, and each
        of those passes through ``handle_async_request``, so a
        redirect to a disallowed host is blocked at the second hop.
        """

        def __init__(
            self,
            enforcer: RuntimePermissionEnforcer,
            inner: Any,
        ) -> None:
            self._enforcer = enforcer
            self._inner = inner

        async def handle_async_request(self, request: Any) -> Any:
            url = str(request.url)
            self._enforcer.assert_url_allowed(url)
            return await self._inner.handle_async_request(request)

        async def aclose(self) -> None:
            await self._inner.aclose()

    # Stable name for repr / tracebacks / logging.
    _AllowlistAsyncTransport.__name__ = "AllowlistAsyncTransport"
    _AllowlistAsyncTransport.__qualname__ = "AllowlistAsyncTransport"
    _AllowlistAsyncTransport.__module__ = __name__

    return _AllowlistAsyncTransport


def _get_allowlist_transport_class() -> type:
    """Return the cached AllowlistAsyncTransport class, building on first call."""
    global _ALLOWLIST_TRANSPORT_CLASS
    if _ALLOWLIST_TRANSPORT_CLASS is None:
        _ALLOWLIST_TRANSPORT_CLASS = _build_allowlist_transport_class()
    return _ALLOWLIST_TRANSPORT_CLASS


def make_allowlist_transport(
    enforcer: RuntimePermissionEnforcer,
    inner: Optional[Any] = None,
) -> Any:
    """
    Build an ``AllowlistAsyncTransport`` instance for ``enforcer``.

    Parameters
    ----------
    enforcer
        The runtime enforcer this transport must obey.
    inner
        The underlying transport to delegate to. When ``None``, a
        default ``httpx.AsyncHTTPTransport()`` is created with TLS
        verification enabled. Tests inject a
        ``httpx.MockTransport`` here to avoid real network I/O.

    Returns
    -------
    httpx.AsyncBaseTransport
        A transport that enforces the allowlist.
    """
    transport_cls = _get_allowlist_transport_class()
    httpx = _import_httpx()

    if inner is None:
        inner = httpx.AsyncHTTPTransport()

    return transport_cls(enforcer=enforcer, inner=inner)


# ---------------------------------------------------------------------------
# Monkey-patch for httpx
# ---------------------------------------------------------------------------


def install_httpx_guard(enforcer: RuntimePermissionEnforcer) -> None:
    """
    Wrap ``httpx.AsyncClient`` so every client the adapter creates is
    subject to the allowlist.

    This is the ONLY way to enforce the allowlist for adapters that
    use SDKs (openai, anthropic, google-genai) whose clients are
    created internally. Those SDKs create ``httpx.AsyncClient``
    instances during their own initialization; patching the class
    before the adapter imports the SDK means every such client is
    wrapped transparently.

    IMPORTANT: Call this BEFORE importing the adapter. Once the
    adapter's SDK has created its client, patching the class has no
    effect on that client. The worker's startup sequence is:

        1. ``capture_stdout_for_protocol()``
        2. ``install_httpx_guard(enforcer)``
        3. import the adapter
        4. instantiate the adapter

    Limitations
    -----------
    * The patch applies to ``httpx.AsyncClient`` only. An adapter
      that uses ``httpx.Client`` (sync) or ``requests`` or raw
      ``socket`` bypasses it.
    * An adapter that already holds a client created BEFORE the
      patch (e.g. at module import time in the SDK) is not
      retroactively wrapped.
    * httpx's own ``AsyncClient`` accepts an explicit ``transport=``
      argument; the patch HONORS it (wraps the provided transport in
      our allowlist) rather than replacing it, so adapters that
      pass a custom transport still get isolation.

    These limitations are documented in the module docstring. They
    are exactly why OS-level sandboxing is required for real
    security guarantees.

    Idempotent
    ----------
    Calling this function twice has no additional effect. The second
    call is a no-op — the wrapped constructor detects that it has
    already been wrapped.
    """
    transport_cls = _get_allowlist_transport_class()
    httpx = _import_httpx()

    original_init = httpx.AsyncClient.__init__

    # Idempotency guard: mark the wrapped constructor.
    if getattr(original_init, "_peldrun_allowlist_wrapped", False):
        return

    def patched_init(self_client, *args, **kwargs):
        # Build or wrap the transport.
        inner_transport = kwargs.get("transport")
        if inner_transport is None:
            inner_transport = httpx.AsyncHTTPTransport()

        # If the caller passed an already-wrapped transport (e.g. our
        # own, from a previous wrap), do not double-wrap.
        if isinstance(inner_transport, transport_cls):
            wrapped = inner_transport
        else:
            wrapped = transport_cls(
                enforcer=enforcer,
                inner=inner_transport,
            )

        kwargs["transport"] = wrapped
        original_init(self_client, *args, **kwargs)

    patched_init._peldrun_allowlist_wrapped = True  # type: ignore[attr-defined]
    httpx.AsyncClient.__init__ = patched_init  # type: ignore[method-assign]

    logger.debug(
        "httpx.AsyncClient patched with allowlist guard "
        "(base_url=%s, provider=%r)",
        enforcer.allowed_base_url,
        enforcer.provider_id,
    )


def __getattr__(name: str) -> Any:
    """
    PEP 562 — lazily expose ``AllowlistAsyncTransport``.

    The class is built on demand because it must inherit from
    ``httpx.AsyncBaseTransport`` — which cannot be established
    without importing httpx. Exposing the name via ``__getattr__``
    keeps ``from peldrun.llm.plugin_host.permissions import
    AllowlistAsyncTransport`` working without forcing an eager httpx
    import at module load.
    """
    if name == "AllowlistAsyncTransport":
        return _get_allowlist_transport_class()
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    # Constants
    "PERM_NETWORK_CONNECT",
    "PERM_SECRETS_READ_CONNECTION",
    "PERM_MODELS_DISCOVER",
    "PERM_LLM_GENERATE",
    "PERM_LLM_STREAM",
    "PERM_USAGE_READ",
    "PERM_WORKSPACE_READ",
    "PERM_WORKSPACE_WRITE",
    "PERM_SHELL_EXECUTE",
    "FORBIDDEN_PERMISSIONS",
    "KNOWN_PERMISSIONS",
    # Exceptions
    "PluginPermissionDeniedError",
    "URLNotAllowedError",
    # Enforcer
    "RuntimePermissionEnforcer",
    # httpx guard
    "AllowlistAsyncTransport",
    "make_allowlist_transport",
    "install_httpx_guard",
]
