import json
import logging
import time

import httpx
from openai import OpenAI

log = logging.getLogger("LiveTranslate.TL")

LANGUAGE_DISPLAY = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "hi": "Hindi",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "el": "Greek",
    "hu": "Hungarian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "he": "Hebrew",
}

DEFAULT_PROMPT = (
    "You are a real-time subtitle translator. Translate {source_lang} into {target_lang}.\n"
    "Rules:\n"
    "- Output ONLY one single best translation, nothing else.\n"
    "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
    "- Keep proper nouns, names, and brand names untranslated.\n"
    "- Pursue faithfulness, expressiveness, and elegance."
)

PROMPT_PRESETS = {
    "daily": (
        "You are a real-time subtitle translator for casual conversation. "
        "Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Keep proper nouns, names, and brand names untranslated.\n"
        "- Use natural, casual, everyday language. Keep it conversational and concise."
    ),
    "esports": (
        "You are a real-time subtitle translator for esports/gaming live streams. "
        "Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Keep player names (IGN), team names, game terms, and brand names untranslated.\n"
        "- Use energetic, concise language appropriate for competitive gaming commentary."
    ),
    "anime": (
        "You are a real-time subtitle translator for anime, movies, and TV shows. "
        "Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Keep character names, place names, and cultural terms untranslated.\n"
        "- Use natural, expressive language that matches the tone and emotion of the dialogue."
    ),
}


def make_openai_client(
    api_base: str, api_key: str, proxy: str = "none", timeout=None
) -> OpenAI:
    kwargs = {"base_url": api_base, "api_key": api_key}
    if timeout is not None:
        kwargs["timeout"] = httpx.Timeout(timeout, connect=5.0)
    if proxy == "system":
        pass
    elif proxy in ("none", "", None):
        kwargs["http_client"] = httpx.Client(trust_env=False)
    else:
        kwargs["http_client"] = httpx.Client(proxy=proxy)
    return OpenAI(**kwargs)


class RepetitionError(Exception):
    """Raised when model output contains repetition loops."""
    pass


class Translator:
    """LLM-based translation using OpenAI-compatible API."""

    def __init__(
        self,
        api_base,
        api_key,
        model,
        target_language="zh",
        max_tokens=256,
        temperature=0.3,
        streaming=True,
        system_prompt=None,
        proxy="none",
        no_system_role=False,
        no_think=False,
        json_response=False,
        timeout=10,
    ):
        self._client = make_openai_client(api_base, api_key, proxy, timeout=timeout)
        self._no_system_role = no_system_role
        self._no_think = no_think
        self._json_response = json_response
        if no_think:
            log.info(f"Translator: no_think enabled for {model}")
        if json_response:
            log.info(f"Translator: json_response enabled for {model}")
        self._model = model
        self._target_language = target_language
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._streaming = streaming
        self._timeout = timeout
        self._system_prompt_template = system_prompt or DEFAULT_PROMPT
        self._context_turns = 0
        self._history = []  # list of (source_text, translated_text)
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0

    @property
    def last_usage(self):
        """(prompt_tokens, completion_tokens) from last translate call."""
        return self._last_prompt_tokens, self._last_completion_tokens

    def set_target_language(self, target_language: str):
        self._target_language = target_language

    def set_timeout(self, timeout: int):
        self._timeout = timeout
        self._client = self._client.copy(timeout=timeout)

    def set_context_turns(self, n: int):
        self._context_turns = n
        if n == 0:
            self._history.clear()

    def clear_history(self):
        self._history.clear()

    def with_target_language(self, target_language: str) -> "Translator":
        """Create a new Translator with a different target language, sharing the same client."""
        t = Translator.__new__(Translator)
        t._client = self._client
        t._no_system_role = self._no_system_role
        t._no_think = self._no_think
        t._json_response = self._json_response
        t._model = self._model
        t._target_language = target_language
        t._max_tokens = self._max_tokens
        t._temperature = self._temperature
        t._streaming = self._streaming
        t._timeout = self._timeout
        t._system_prompt_template = self._system_prompt_template
        t._context_turns = 0
        t._history = []
        t._last_prompt_tokens = 0
        t._last_completion_tokens = 0
        return t

    def _build_system_prompt(self, source_lang):
        src = LANGUAGE_DISPLAY.get(source_lang, source_lang)
        tgt = LANGUAGE_DISPLAY.get(self._target_language, self._target_language)
        try:
            prompt = self._system_prompt_template.format(
                source_lang=src,
                target_lang=tgt,
            )
        except (KeyError, IndexError, ValueError) as e:
            log.warning(f"Bad prompt template, falling back to default: {e}")
            prompt = DEFAULT_PROMPT.format(source_lang=src, target_lang=tgt)
        if self._json_response:
            prompt += '\nRespond in JSON format: {"t": "translated text"}'
        return prompt

    def _build_messages(self, system_prompt, text):
        if self._no_system_role:
            msgs = [{"role": "user", "content": f"{system_prompt}\n{text}"}]
        else:
            msgs = [{"role": "system", "content": system_prompt}]
            # Append recent history as context
            if self._context_turns > 0 and self._history:
                for src, tgt in self._history[-self._context_turns:]:
                    msgs.append({"role": "user", "content": src})
                    msgs.append({"role": "assistant", "content": tgt})
            msgs.append({"role": "user", "content": text})
        return msgs

    def _append_history(self, text, result):
        if self._context_turns > 0 and result:
            self._history.append((text, result))
            max_keep = self._context_turns + 2
            if len(self._history) > max_keep:
                self._history = self._history[-self._context_turns:]

    def translate(self, text: str, source_language: str = "en"):
        system_prompt = self._build_system_prompt(source_language)
        if self._streaming:
            result = self._translate_streaming(system_prompt, text)
        else:
            result = self._translate_sync(system_prompt, text)
        if self._check_repetition(result):
            raise RepetitionError(result)
        self._append_history(text, result)
        return result

    def translate_iter(self, text: str, source_language: str = "en"):
        """Generator that yields accumulated partial text, then final result.

        Non-streaming or json_response mode: yields once with the final result.
        Streaming mode: yields partial accumulated text as chunks arrive.
        The final yielded value is always the complete translation.
        Caller should use the last yielded value as the final result.
        """
        system_prompt = self._build_system_prompt(source_language)
        if not self._streaming:
            result = self._translate_sync(system_prompt, text)
            self._append_history(text, result)
            yield result
            return

        # Streaming path
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        base_kwargs = dict(
            model=self._model,
            messages=self._build_messages(system_prompt, text),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )
        if self._no_think:
            base_kwargs["extra_body"] = {"enable_thinking": False}
        if self._json_response:
            base_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"t": {"type": "string"}},
                        "required": ["t"],
                        "additionalProperties": False,
                    },
                },
            }
        try:
            stream = self._client.chat.completions.create(
                **base_kwargs,
                stream_options={"include_usage": True},
            )
        except Exception:
            stream = self._client.chat.completions.create(**base_kwargs)

        deadline = time.monotonic() + self._timeout
        chunks = []
        for chunk in stream:
            if time.monotonic() > deadline:
                stream.close()
                raise TimeoutError(
                    f"Translation exceeded {self._timeout}s total timeout"
                )
            if hasattr(chunk, "usage") and chunk.usage:
                self._last_prompt_tokens = chunk.usage.prompt_tokens or 0
                self._last_completion_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
                    if not self._json_response:
                        yield "".join(chunks)
        result = "".join(chunks).strip()
        if self._json_response:
            result = self._extract_json_translation(result)
        if self._check_repetition(result):
            raise RepetitionError(result)
        self._append_history(text, result)
        yield result

    def _extract_json_translation(self, raw: str) -> str:
        """Extract translation from JSON response, fallback to raw text."""
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "t" in data:
                return data["t"]
        except (json.JSONDecodeError, TypeError):
            pass
        return raw

    @staticmethod
    def _check_repetition(text: str) -> bool:
        """Detect repetition loops in model output."""
        if not text or len(text) < 40:
            return False
        for plen in range(8, len(text) // 2 + 1):
            if text[plen:plen * 2] == text[:plen]:
                return True
        return False

    def _translate_sync(self, system_prompt, text):
        kwargs = dict(
            model=self._model,
            messages=self._build_messages(system_prompt, text),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        if self._no_think:
            kwargs["extra_body"] = {"enable_thinking": False}
        if self._json_response:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"t": {"type": "string"}},
                        "required": ["t"],
                        "additionalProperties": False,
                    },
                },
            }
        resp = self._client.chat.completions.create(**kwargs)
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        if resp.usage:
            self._last_prompt_tokens = resp.usage.prompt_tokens or 0
            self._last_completion_tokens = resp.usage.completion_tokens or 0
        result = resp.choices[0].message.content.strip()
        if self._json_response:
            result = self._extract_json_translation(result)
        return result

    def _translate_streaming(self, system_prompt, text):
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        base_kwargs = dict(
            model=self._model,
            messages=self._build_messages(system_prompt, text),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )
        if self._no_think:
            base_kwargs["extra_body"] = {"enable_thinking": False}
        if self._json_response:
            base_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"t": {"type": "string"}},
                        "required": ["t"],
                        "additionalProperties": False,
                    },
                },
            }
        try:
            stream = self._client.chat.completions.create(
                **base_kwargs,
                stream_options={"include_usage": True},
            )
        except Exception:
            stream = self._client.chat.completions.create(**base_kwargs)

        deadline = time.monotonic() + self._timeout
        chunks = []
        for chunk in stream:
            if time.monotonic() > deadline:
                stream.close()
                raise TimeoutError(
                    f"Translation exceeded {self._timeout}s total timeout"
                )
            if hasattr(chunk, "usage") and chunk.usage:
                self._last_prompt_tokens = chunk.usage.prompt_tokens or 0
                self._last_completion_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
        result = "".join(chunks).strip()
        if self._json_response:
            result = self._extract_json_translation(result)
        return result
