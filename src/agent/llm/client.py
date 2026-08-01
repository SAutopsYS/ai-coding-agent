"""Provider-agnostic LLM chat client (OpenAI, Anthropic, Gemini, mock)."""

from __future__ import annotations

import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.llm.schemas import EditInstructions, FileEdit

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """One message in a chat completion request."""

    role: str
    content: str


@dataclass
class LLMClientConfig:
    """Configuration for the LLM client."""

    provider: str = "gemini"
    model: str = "gemini-3.6-flash"
    temperature: float = 0.2
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 90.0
    extra: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Protocol for concrete chat providers."""

    def complete(self, messages: list[ChatMessage], *, model: str, temperature: float) -> str:
        """Return assistant text for the given chat messages."""
        ...


def _http_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM connection failed: {exc}") from exc
    return json.loads(body)


class OpenAIProvider:
    """OpenAI Chat Completions API (also works with compatible gateways)."""

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 90.0) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout

    def complete(self, messages: list[ChatMessage], *, model: str, temperature: float) -> str:
        """Call OpenAI chat completions and return the assistant message text."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = _http_json(url, payload, headers, timeout=self.timeout)
        return data["choices"][0]["message"]["content"]


class AnthropicProvider:
    """Anthropic Messages API."""

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 90.0) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.timeout = timeout

    def complete(self, messages: list[ChatMessage], *, model: str, temperature: float) -> str:
        """Call Anthropic Messages API and return concatenated text blocks."""
        system_parts = [m.content for m in messages if m.role == "system"]
        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in {"user", "assistant"}
        ]
        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Return an empty edits JSON object."}]
        url = f"{self.base_url}/v1/messages"
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 4096,
            "messages": chat_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = _http_json(url, payload, headers, timeout=self.timeout)
        blocks = data.get("content") or []
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        return "\n".join(texts)


class GeminiProvider:
    """Google Gemini provider via the official Google GenAI Python SDK.

    Same ``complete(messages, model=..., temperature=...)`` interface as other providers.
    Requests structured JSON compatible with ``EditInstructions``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 90.0,
        *,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = self._create_client(api_key, base_url)

    @staticmethod
    def _create_client(api_key: str, base_url: str | None) -> Any:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for the Gemini provider. "
                "Install with: pip install google-genai"
            ) from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["http_options"] = {"base_url": base_url, "timeout": int(90_000)}
        logger.info("Initializing Google GenAI client for Gemini provider")
        return genai.Client(**kwargs)

    def complete(self, messages: list[ChatMessage], *, model: str, temperature: float) -> str:
        """Generate structured JSON text with the Google GenAI SDK."""
        from google.genai import errors, types

        system_parts = [m.content for m in messages if m.role == "system"]
        contents: list[types.Content] = []
        for message in messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message.content)],
                )
            )
        if not contents:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Return EditInstructions JSON.")],
                )
            ]

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=EditInstructions,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Gemini generate_content model=%s attempt=%d/%d",
                    model,
                    attempt,
                    self.max_retries,
                )
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty response")
                logger.debug("Gemini response length=%d", len(text))
                return text
            except errors.ClientError as exc:
                last_error = exc
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if code in {408, 429} and attempt < self.max_retries:
                    self._sleep(attempt)
                    continue
                logger.error("Gemini client error (code=%s): %s", code, exc)
                raise RuntimeError(f"Gemini client error: {exc}") from exc
            except errors.ServerError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(attempt)
                    continue
                logger.error("Gemini server error after retries: %s", exc)
                raise RuntimeError(f"Gemini server error: {exc}") from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(attempt)
                    continue
                logger.error("Gemini connection failure after retries: %s", exc)
                raise RuntimeError(f"Gemini connection failed: {exc}") from exc
            except Exception as exc:  # noqa: BLE001 - normalize SDK surprises
                last_error = exc
                message = str(exc).lower()
                transient = any(
                    token in message
                    for token in ("timeout", "temporarily", "unavailable", "429", "500", "503")
                )
                if transient and attempt < self.max_retries:
                    self._sleep(attempt)
                    continue
                logger.error("Gemini request failed: %s", exc)
                raise RuntimeError(f"Gemini request failed: {exc}") from exc

        raise RuntimeError(f"Gemini request failed after retries: {last_error}")

    def _sleep(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
        logger.warning("Gemini transient failure; retrying in %.1fs", delay)
        time.sleep(delay)


class MockProvider:
    """Offline provider that emits structured JSON from the latest user payload.

    Used for tests and demo runs without API keys. It does not hardcode a
    specific repository name; it inspects provided file excerpts and intents.
    """

    def complete(self, messages: list[ChatMessage], *, model: str, temperature: float) -> str:
        """Return deterministic EditInstructions JSON from prompt file excerpts."""
        del model, temperature
        user_blobs = [m.content for m in messages if m.role == "user"]
        blob = "\n".join(user_blobs)
        edits = _heuristic_edits_from_prompt(blob)
        payload = EditInstructions(
            thought="Mock provider generated deterministic structured edits from plan/context.",
            edits=edits,
            done=True,
            notes="Generated by mock provider (no external LLM call).",
        )
        return payload.model_dump_json()


def _heuristic_edits_from_prompt(blob: str) -> list[FileEdit]:
    """Derive generic organize/search edits from file excerpts in the prompt."""
    edits: list[FileEdit] = []
    lower = blob.lower()
    wants_organize = any(k in lower for k in ("organize", "organise", "tag", "tags"))
    wants_search = "search" in lower or "filter" in lower

    files = _extract_file_blocks(blob)
    for path, content in files.items():
        path_l = path.lower()
        if wants_organize and ("model" in path_l or "schema" in path_l):
            edit = _model_tags_edit(path, content)
            if edit:
                edits.append(edit)
        if wants_search and ("controller" in path_l or "handler" in path_l or "service" in path_l):
            edit = _controller_search_edit(path, content)
            if edit:
                edits.append(edit)
        if wants_search and ("route" in path_l or "router" in path_l):
            edit = _routes_search_edit(path, content)
            if edit:
                edits.append(edit)
        if wants_organize and ("controller" in path_l or "handler" in path_l):
            edits.extend(_controller_tags_edits(path, content))
        if path_l.endswith("readme.md") or path_l.endswith("readme"):
            edit = _readme_edit(path, content, wants_organize=wants_organize, wants_search=wants_search)
            if edit:
                edits.append(edit)

    return edits


_FILE_BLOCK_RE = re.compile(
    r"<<<FILE path=\"(?P<path>[^\"]+)\">>>\n(?P<content>.*?)\n<<<END_FILE>>>",
    re.DOTALL,
)


def _extract_file_blocks(blob: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for match in _FILE_BLOCK_RE.finditer(blob):
        path = match.group("path").strip()
        content = match.group("content")
        files[path] = content
    return files


def _model_tags_edit(path: str, content: str) -> FileEdit | None:
    if "tags" in content:
        return None
    old = None
    new = None
    if "mongoose.Schema" in content or "Schema({" in content:
        # Prefer [ \t] over \\s so the match stops before the newline.
        match = re.search(r"(content[ \t]*:[ \t]*String[ \t]*,?)", content)
        if match:
            old = match.group(1).rstrip()
            new = old.rstrip(",") + ",\n    tags: [String]"
        else:
            match = re.search(r"(title[ \t]*:[ \t]*String[ \t]*,?)", content)
            if match:
                old = match.group(1).rstrip()
                new = old.rstrip(",") + ",\n    tags: [String]"
    if old and new and old != new:
        return FileEdit(
            path=path,
            action="replace",
            old_string=old,
            new_string=new,
            reason="Add optional tags field so notes can be organized.",
        )
    return None


def _controller_tags_edits(path: str, content: str) -> list[FileEdit]:
    edits: list[FileEdit] = []
    pattern = re.compile(
        r"title:\s*req\.body\.title(?:\s*\|\|[^,\n]+)?\s*,\s*\n\s*content:\s*req\.body\.content"
    )
    for match in pattern.finditer(content):
        old = match.group(0)
        if "tags:" in old:
            continue
        new = old + ",\n        tags: Array.isArray(req.body.tags) ? req.body.tags : []"
        edits.append(
            FileEdit(
                path=path,
                action="replace",
                old_string=old,
                new_string=new,
                reason="Persist optional tags on create/update payloads.",
            )
        )
    return edits


def _controller_search_edit(path: str, content: str) -> FileEdit | None:
    if "exports.search" in content or "def search" in content:
        return None
    if "exports.findAll" in content and "Note.find" in content:
        old = """// Retrieve and return all notes from the database.
exports.findAll = (req, res) => {
    Note.find()
    .then(notes => {
        res.send(notes);
    }).catch(err => {
        res.status(500).send({
            message: err.message || "Some error occurred while retrieving notes."
        });
    });
};"""
        if old not in content:
            marker = "// Find a single note with a noteId"
            if marker not in content:
                return None
            search_fn = '''
// Search notes by title, content, and tags.
exports.search = (req, res) => {
    const q = (req.query.q || "").toString().trim();
    const tag = (req.query.tag || "").toString().trim();
    const criteria = {};
    if (q) {
        criteria.$or = [
            { title: { $regex: q, $options: "i" } },
            { content: { $regex: q, $options: "i" } },
            { tags: { $regex: q, $options: "i" } }
        ];
    }
    if (tag) {
        criteria.tags = tag;
    }
    Note.find(criteria)
    .then(notes => {
        res.send(notes);
    }).catch(err => {
        res.status(500).send({
            message: err.message || "Some error occurred while searching notes."
        });
    });
};

'''
            return FileEdit(
                path=path,
                action="replace",
                old_string=marker,
                new_string=search_fn + marker,
                reason="Add search handler supporting q and tag query params.",
            )
        new = """// Retrieve and return all notes from the database.
// Optional filters: ?q=<text>&tag=<tag>
exports.findAll = (req, res) => {
    const q = (req.query.q || "").toString().trim();
    const tag = (req.query.tag || "").toString().trim();
    const criteria = {};
    if (q) {
        criteria.$or = [
            { title: { $regex: q, $options: "i" } },
            { content: { $regex: q, $options: "i" } },
            { tags: { $regex: q, $options: "i" } }
        ];
    }
    if (tag) {
        criteria.tags = tag;
    }
    Note.find(Object.keys(criteria).length ? criteria : {})
    .then(notes => {
        res.send(notes);
    }).catch(err => {
        res.status(500).send({
            message: err.message || "Some error occurred while retrieving notes."
        });
    });
};

// Search notes by title, content, and tags.
exports.search = (req, res) => {
    exports.findAll(req, res);
};"""
        return FileEdit(
            path=path,
            action="replace",
            old_string=old,
            new_string=new,
            reason="Add search/filter support on list/search handlers.",
        )
    return None


def _routes_search_edit(path: str, content: str) -> FileEdit | None:
    if "/search" in content or "notes.search" in content:
        return None
    if "app.get('/notes'" in content or 'app.get("/notes"' in content:
        old = "    // Retrieve all Notes\n    app.get('/notes', notes.findAll);"
        if old in content:
            new = (
                "    // Retrieve all Notes (supports optional ?q=&tag= filters)\n"
                "    app.get('/notes', notes.findAll);\n\n"
                "    // Search notes by query/tag\n"
                "    app.get('/notes/search', notes.search);"
            )
            return FileEdit(
                path=path,
                action="replace",
                old_string=old,
                new_string=new,
                reason="Expose GET /notes/search before parameterized routes.",
            )

        match = re.search(r"(app\.get\(['\"]\/notes\/:)", content)
        if match:
            insert_at = match.start()
            snippet = (
                "    // Search notes by query/tag\n"
                "    app.get('/notes/search', notes.search);\n\n    "
            )
            old_prefix = content[insert_at : insert_at + len(match.group(1))]
            return FileEdit(
                path=path,
                action="replace",
                old_string=old_prefix,
                new_string=snippet + old_prefix,
                reason="Register search route before :id route.",
            )
    return None


def _readme_edit(
    path: str,
    content: str,
    *,
    wants_organize: bool,
    wants_search: bool,
) -> FileEdit | None:
    if "Search notes" in content or "`/notes/search`" in content:
        return None
    addition = "\n\n## Organization & Search\n\n"
    if wants_organize:
        addition += (
            "- Notes accept an optional `tags` string array on create/update.\n"
        )
    if wants_search:
        addition += (
            "- `GET /notes?q=<text>&tag=<tag>` filters notes.\n"
            "- `GET /notes/search?q=<text>&tag=<tag>` searches title, content, and tags.\n"
        )
    if not wants_organize and not wants_search:
        return None
    old = content[-80:] if len(content) > 80 else content
    if not old:
        return FileEdit(
            path=path,
            action="write",
            content=addition.strip() + "\n",
            reason="Document organization and search behavior.",
        )
    return FileEdit(
        path=path,
        action="replace",
        old_string=old,
        new_string=old + addition,
        reason="Document organization and search behavior.",
    )


class LLMClient:
    """Thin wrapper around a chat-completion provider."""

    def __init__(
        self,
        config: LLMClientConfig | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.config = config or LLMClientConfig()
        self._provider = provider or self._build_provider(self.config)

    @staticmethod
    def _build_provider(config: LLMClientConfig) -> LLMProvider:
        name = (config.provider or "openai").strip().lower()
        if name == "mock":
            logger.info("Using mock LLM provider")
            return MockProvider()
        if not config.api_key:
            raise ValueError(
                f"LLM provider '{name}' requires AGENT_LLM_API_KEY (or config api_key)."
            )
        if name == "openai":
            return OpenAIProvider(config.api_key, config.base_url, config.timeout_seconds)
        if name == "anthropic":
            return AnthropicProvider(config.api_key, config.base_url, config.timeout_seconds)
        if name in {"gemini", "google"}:
            max_retries = int(config.extra.get("max_retries", 3))
            return GeminiProvider(
                config.api_key,
                config.base_url,
                config.timeout_seconds,
                max_retries=max_retries,
            )
        raise ValueError(f"Unsupported LLM provider: {name}")

    def complete(self, messages: list[ChatMessage]) -> str:
        """Return the assistant message content for the given messages."""
        logger.info(
            "Calling LLM provider=%s model=%s messages=%d",
            self.config.provider,
            self.config.model,
            len(messages),
        )
        try:
            text = self._provider.complete(
                messages,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        except Exception:
            logger.exception(
                "LLM provider=%s model=%s failed",
                self.config.provider,
                self.config.model,
            )
            raise
        logger.debug("LLM raw response length=%d", len(text or ""))
        return text
