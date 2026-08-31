"""LLM-based fragment improvers: DeepSeek, Anthropic, and heuristic fallback.

When a fragment falls below the quality threshold, these improvers
send it to an LLM for enrichment and return the improved version.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from memory_fragments.models import Fragment, FragmentMetadata

# ---------------------------------------------------------------------------
# Shared prompt
# ---------------------------------------------------------------------------

IMPROVE_PROMPT = """You are a fragment quality enhancer. Your task is to improve the following
knowledge fragment so it is more informative, precise, and well-structured.

Current quality score: {quality:.2f} (target: ≥ 0.80)

Fragment content:
{content}

Fragment topic: {topic}
Fragment tags: {tags}

Return a JSON object with these fields:
- "content": the improved, expanded version of the fragment
- "quality": a new quality score between 0.0 and 1.0
- "topic": the topic (improved if needed)
- "tags": list of relevant tags
- "improvement_notes": brief description of what you improved

Only respond with valid JSON, no preamble."""


# ---------------------------------------------------------------------------
# DeepSeek Improver (OpenAI-compatible API)
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


class DeepSeekFragmentImprover:
    """Uses DeepSeek (OpenAI-compatible API) to improve low-quality fragments.

    Requires ``pip install openai`` and the ``DEEPSEEK_API_KEY``
    environment variable.

    Usage::

        improver = DeepSeekFragmentImprover(api_key="sk-...")
        improved = improver(low_quality_fragment)

    Or via environment variable::

        export DEEPSEEK_API_KEY="sk-..."
        improver = DeepSeekFragmentImprover()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 1,
    ) -> None:
        if not _HAS_OPENAI:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError(
                "DeepSeek API key required. Pass api_key= or set DEEPSEEK_API_KEY env var."
            )
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._model = model
        self._max_retries = max_retries

    def __call__(self, fragment: Fragment) -> Optional[Fragment]:
        """Improve a fragment via DeepSeek. Returns improved copy or None."""
        prompt = IMPROVE_PROMPT.format(
            quality=fragment.metadata.quality,
            content=fragment.content,
            topic=fragment.metadata.topic or "untitled",
            tags=", ".join(fragment.metadata.tags) or "none",
        )

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.choices[0].message.content or ""
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw
                    raw = raw.rsplit("```", 1)[0] if "```" in raw else raw
                data: Dict[str, Any] = json.loads(raw.strip())

                improved = Fragment(
                    fragment_id=f"{fragment.fragment_id}_improved",
                    content=data.get("content", fragment.content),
                    metadata=FragmentMetadata(
                        topic=data.get("topic", fragment.metadata.topic),
                        quality=float(data.get("quality", fragment.metadata.quality)),
                        tags=data.get("tags", fragment.metadata.tags),
                        source=fragment.metadata.source,
                    ),
                    conditions=fragment.conditions,
                    parents=[fragment.fragment_id],
                )
                return improved

            except Exception:
                if attempt >= self._max_retries:
                    return None
                continue

        return None


# ---------------------------------------------------------------------------
# Anthropic Improver
# ---------------------------------------------------------------------------

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


class AnthropicFragmentImprover:
    """Uses the Anthropic API (Claude) to improve low-quality fragments.

    Requires ``pip install anthropic`` and the ``ANTHROPIC_API_KEY``
    environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 1,
    ) -> None:
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_retries = max_retries

    def __call__(self, fragment: Fragment) -> Optional[Fragment]:
        """Improve a fragment via Claude. Returns improved copy or None."""
        prompt = IMPROVE_PROMPT.format(
            quality=fragment.metadata.quality,
            content=fragment.content,
            topic=fragment.metadata.topic or "untitled",
            tags=", ".join(fragment.metadata.tags) or "none",
        )

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text if response.content else ""
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw
                    raw = raw.rsplit("```", 1)[0] if "```" in raw else raw
                data: Dict[str, Any] = json.loads(raw.strip())

                improved = Fragment(
                    fragment_id=f"{fragment.fragment_id}_improved",
                    content=data.get("content", fragment.content),
                    metadata=FragmentMetadata(
                        topic=data.get("topic", fragment.metadata.topic),
                        quality=float(data.get("quality", fragment.metadata.quality)),
                        tags=data.get("tags", fragment.metadata.tags),
                        source=fragment.metadata.source,
                    ),
                    conditions=fragment.conditions,
                    parents=[fragment.fragment_id],
                )
                return improved

            except Exception:
                if attempt >= self._max_retries:
                    return None
                continue

        return None


# ---------------------------------------------------------------------------
# Heuristic Improver (no API needed — for testing / offline)
# ---------------------------------------------------------------------------


class HeuristicImprover:
    """Simple rule-based improvver for testing (no external API needed).

    Boosts quality by enriching content and metadata. Useful as a
    development stand-in for LLM-based improvers.
    """

    def __init__(self, boost: float = 0.15) -> None:
        self._boost = boost

    def __call__(self, fragment: Fragment) -> Optional[Fragment]:
        improved = Fragment(
            fragment_id=f"{fragment.fragment_id}_improved",
            content=f"{fragment.content} [Enhanced with additional context and structured formatting for clarity.]",
            metadata=FragmentMetadata(
                topic=fragment.metadata.topic,
                quality=min(fragment.metadata.quality + self._boost, 0.95),
                tags=list(set(fragment.metadata.tags + ["enhanced", "auto_improved"])),
                source=fragment.metadata.source,
            ),
            conditions=fragment.conditions,
            parents=[fragment.fragment_id],
        )
        return improved if improved.metadata.quality >= 0.80 else None
