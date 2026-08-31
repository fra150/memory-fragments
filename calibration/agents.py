"""Agent implementations for 3-agent quality voting calibration.

Provides agent wrappers that interface with local LLMs (via subprocess, HTTP API)
or simulated mock agents for testing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import request, error

from memory_fragments.models import Fragment
from memory_fragments.models.quality import QualitySource, QualityEvaluation
from memory_fragments.calibration.validator import AgentEvaluator


# ---------------------------------------------------------------------------
# Prompt template for quality evaluator agents
# ---------------------------------------------------------------------------

QUALITY_EVALUATOR_PROMPT = """You are a quality evaluator agent. Your task is to assess the quality of a knowledge fragment.

Rate the fragment on these criteria (0.0 - 1.0 scale):
1. **Factual accuracy** — does it contain correct information?
2. **Completeness** — does it provide sufficient detail?
3. **Clarity** — is it well-written and understandable?
4. **Usefulness** — is it actionable or informative?

Fragment topic: {topic}
Fragment content:
{content}

Return ONLY a JSON object with:
- "score": float (0.0 to 1.0, overall quality score)
- "confidence": float (0.0 to 1.0, how confident you are in this score)
- "reasoning": str (brief explanation for the score)

No preamble or extra text."""


# ---------------------------------------------------------------------------
# Agent configurations
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration for a single quality evaluation agent."""

    name: str  # Agent identifier, e.g. "agent-alpha"
    model_id: str  # Model identifier, e.g. "phi-3-mini-4k"
    model_version: str = ""  # Optional version string
    temperature: float = 0.3  # Lower = more consistent scoring
    max_tokens: int = 256
    timeout_seconds: float = 10.0

    # Quality source assigned to this agent's evaluations
    quality_source: QualitySource = QualitySource.LLM_SELF_REPORTED


# ---------------------------------------------------------------------------
# Mock agent (for testing without real LLMs)
# ---------------------------------------------------------------------------


class MockQualityAgent:
    """Deterministic mock agent for testing the calibration framework.

    Assigns scores based on simple heuristics (no LLM call).
    Useful for unit tests and calibration pipeline validation.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.call_count = 0

    def __call__(self, fragment: Fragment) -> Tuple[float, Dict[str, Any]]:
        self.call_count += 1
        content = fragment.content
        quality = fragment.metadata.quality

        # Heuristic scoring
        word_count = len(content.split())

        # Penalize very short fragments
        if word_count < 5:
            score = max(0.0, quality - 0.3)
        elif word_count < 15:
            score = quality  # Neutral on moderate length
        else:
            score = min(1.0, quality + 0.1)  # Slight bonus for detail

        # Check for "junk" indicators (negation words, contradictions in key terms)
        junk_indicators = ["wrong", "fake", "actually", "but actually", "incorrectly"]
        text_lower = content.lower()
        for indicator in junk_indicators:
            if indicator in text_lower:
                score = max(0.0, score - 0.4)
                break

        # Add small random noise to simulate agent variation (±0.05)
        import random

        rng = random.Random(hash(content) + hash(self.config.name))
        noise = (rng.random() - 0.5) * 0.1
        score = max(0.0, min(1.0, score + noise))

        return round(score, 4), {
            "agent": self.config.name,
            "word_count": word_count,
            "call_count": self.call_count,
        }


# ---------------------------------------------------------------------------
# Subprocess-based local agent (for llama.cpp CLI)
# ---------------------------------------------------------------------------


class LocalModelAgent:
    """Agent that calls a local LLM via subprocess (llama.cpp CLI).

    Expects ``llama-cli`` (or the given command) to be available on PATH.
    Falls back to mock behavior when the command is not found.

    Usage::

        agent = LocalModelAgent(
            AgentConfig(name="agent-phi3", model_id="phi-3-mini-4k"),
            command="llama-cli",
            model_path="models/phi-3-mini-4k.Q4_K_M.gguf",
        )
        score, meta = agent(fragment)
    """

    def __init__(
        self,
        config: AgentConfig,
        command: str = "llama-cli",
        model_path: str = "",
        fallback_to_mock: bool = True,
    ) -> None:
        self.config = config
        self.command = command
        self.model_path = model_path
        self._fallback = fallback_to_mock
        self._mock: Optional[MockQualityAgent] = None
        self._available: Optional[bool] = None  # Lazy check

    def __call__(self, fragment: Fragment) -> Tuple[float, Dict[str, Any]]:
        if self._available is None:
            self._available = self._check_available()

        if not self._available:
            if self._fallback:
                if self._mock is None:
                    self._mock = MockQualityAgent(self.config)
                return self._mock(fragment)
            raise RuntimeError(
                f"Local model command '{self.command}' not available "
                f"and fallback_to_mock=False"
            )

        prompt = QUALITY_EVALUATOR_PROMPT.format(
            topic=fragment.metadata.topic or "general",
            content=fragment.content,
        )

        try:
            result = subprocess.run(
                [self.command, "--prompt", prompt, "--model", self.model_path],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            return self._parse_output(result.stdout)
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            if self._fallback:
                if self._mock is None:
                    self._mock = MockQualityAgent(self.config)
                return self._mock(fragment)
            raise RuntimeError(f"Local model call failed: {e}") from e

    def _check_available(self) -> bool:
        """Check if the command is available on PATH and model file exists."""
        try:
            subprocess.run([self.command, "--version"], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        if self.model_path and not os.path.exists(os.path.expanduser(self.model_path)):
            return False
        return True

    def _parse_output(self, output: str) -> Tuple[float, Dict[str, Any]]:
        """Parse JSON from model output."""
        try:
            output = output.strip()
            if "```" in output:
                # Strip markdown code fences
                parts = output.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") or part.startswith("json"):
                        output = part.lstrip("json").strip()
                        break
            data = json.loads(output)
            score = float(data.get("score", 0.5))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")
            return max(0.0, min(1.0, score)), {
                "confidence": confidence,
                "reasoning": reasoning,
                "agent": self.config.name,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback: extract any number from output
            score = 0.5
            for line in output.split("\n"):
                if "score" in line.lower():
                    import re

                    nums = re.findall(r"0\.\d+|1\.0", line)
                    if nums:
                        score = float(nums[0])
                        break
            return max(0.0, min(1.0, score)), {
                "parse_failed": True,
                "raw_output": output[:200],
                "agent": self.config.name,
            }


# ---------------------------------------------------------------------------
# Ollama-based agent
# ---------------------------------------------------------------------------


class OllamaQualityAgent:
    """Agent that calls a model via Ollama API.

    Requires Ollama running locally (default: http://localhost:11434).

    Usage::

        agent = OllamaQualityAgent(
            AgentConfig(name="agent-gemma", model_id="gemma:2b"),
            model="gemma:2b",
        )
        score, meta = agent(fragment)
    """

    def __init__(
        self,
        config: AgentConfig,
        ollama_model: str = "gemma:2b",
        base_url: str = "http://localhost:11434",
        fallback_to_mock: bool = True,
    ) -> None:
        self.config = config
        self.ollama_model = ollama_model
        self.base_url = base_url.rstrip("/")
        self._fallback = fallback_to_mock
        self._mock: Optional[MockQualityAgent] = None
        self._available: Optional[bool] = None

    def __call__(self, fragment: Fragment) -> Tuple[float, Dict[str, Any]]:
        if self._available is None:
            self._available = self._check_available()

        if not self._available:
            if self._fallback:
                if self._mock is None:
                    self._mock = MockQualityAgent(self.config)
                return self._mock(fragment)
            raise RuntimeError(
                f"Ollama at {self.base_url} not available "
                f"and fallback_to_mock=False"
            )

        prompt = QUALITY_EVALUATOR_PROMPT.format(
            topic=fragment.metadata.topic or "general",
            content=fragment.content,
        )

        payload = json.dumps(
            {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
        ).encode("utf-8")

        try:
            req = request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
                return self._parse_output(response_data.get("response", ""))
        except (
            error.URLError,
            error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
        ) as e:
            if self._fallback:
                if self._mock is None:
                    self._mock = MockQualityAgent(self.config)
                return self._mock(fragment)
            raise RuntimeError(f"Ollama call failed: {e}") from e

    def _check_available(self) -> bool:
        try:
            req = request.Request(f"{self.base_url}/api/tags")
            with request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                return self.ollama_model in models
        except Exception:
            return False

    def _parse_output(self, output: str) -> Tuple[float, Dict[str, Any]]:
        """Parse JSON from Ollama model output."""
        try:
            output = output.strip()
            if "```" in output:
                parts = output.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") or part.startswith("json"):
                        output = part.lstrip("json").strip()
                        break
            data = json.loads(output)
            score = float(data.get("score", 0.5))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")
            return max(0.0, min(1.0, score)), {
                "confidence": confidence,
                "reasoning": reasoning[:200],
                "agent": self.config.name,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            score = 0.5
            import re

            for line in output.split("\n"):
                if "score" in line.lower():
                    nums = re.findall(r"0\.\d+|1\.0", line)
                    if nums:
                        score = float(nums[0])
                        break
            return max(0.0, min(1.0, score)), {
                "parse_failed": True,
                "raw_output": output[:200],
                "agent": self.config.name,
            }


# ---------------------------------------------------------------------------
# OpenAI-compatible API agent (for remote API models)
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


class OpenAIApiAgent:
    """Agent that calls an OpenAI-compatible API (remote or local proxy).

    Works with any OpenAI-compatible endpoint: OpenAI, DeepSeek, Together AI,
    or local proxy (e.g., llama.cpp server, vLLM).

    Usage::

        agent = OpenAIApiAgent(
            AgentConfig(name="agent-alpha", model_id="gpt-4o-mini"),
            api_key="sk-...",
            model="gpt-4o-mini",
        )
        score, meta = agent(fragment)
    """

    def __init__(
        self,
        config: AgentConfig,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        fallback_to_mock: bool = True,
    ) -> None:
        self.config = config
        self.model = model
        self._fallback = fallback_to_mock
        self._mock: Optional[MockQualityAgent] = None

        if _HAS_OPENAI:
            self._client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
                base_url=base_url,
            )
        else:
            self._client = None

    def __call__(self, fragment: Fragment) -> Tuple[float, Dict[str, Any]]:
        if self._client is None:
            if self._fallback:
                if self._mock is None:
                    self._mock = MockQualityAgent(self.config)
                return self._mock(fragment)
            raise RuntimeError(
                "OpenAI client not available (install openai package) "
                "and fallback_to_mock=False"
            )

        prompt = QUALITY_EVALUATOR_PROMPT.format(
            topic=fragment.metadata.topic or "general",
            content=fragment.content,
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_response(raw)
        except Exception as e:
            if self._fallback:
                if self._mock is None:
                    self._mock = MockQualityAgent(self.config)
                return self._mock(fragment)
            raise RuntimeError(f"API call failed: {e}") from e

    def _parse_response(self, raw: str) -> Tuple[float, Dict[str, Any]]:
        raw = raw.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") or part.startswith("json"):
                    raw = part.lstrip("json").strip()
                    break
        try:
            data = json.loads(raw)
            score = float(data.get("score", 0.5))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")
            return max(0.0, min(1.0, score)), {
                "confidence": confidence,
                "reasoning": reasoning[:200],
                "agent": self.config.name,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            score = 0.5
            import re

            nums = re.findall(r"0\.\d+|1\.0", raw)
            if nums:
                score = float(nums[0])
            return max(0.0, min(1.0, score)), {
                "parse_failed": True,
                "raw_output": raw[:200],
                "agent": self.config.name,
            }


# ---------------------------------------------------------------------------
# Factory and convenience functions
# ---------------------------------------------------------------------------


def create_mock_agents(
    names: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, AgentEvaluator]:
    """Create a set of deterministic mock agents for testing.

    Creates 3 agents with different deterministic behaviors to simulate
    realistic disagreement patterns:

    - ``"agent-strict"`` — tends to score lower (conservative)
    - ``"agent-balanced"`` — neutral, matches claimed quality
    - ``"agent-generous"`` — tends to score higher

    Args:
        names: Custom agent names (default: ``["agent-strict", "agent-balanced",
               "agent-generous"]``).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping agent name to evaluator function.
    """
    if names is None:
        names = ["agent-strict", "agent-balanced", "agent-generous"]

    biases = {"agent-strict": -0.15, "agent-balanced": 0.0, "agent-generous": 0.15}

    agents: Dict[str, AgentEvaluator] = {}
    for name in names:
        bias = biases.get(name, 0.0)
        config = AgentConfig(name=name, model_id=name)
        mock = MockQualityAgent(config)

        # Wrap to apply bias — use default-arg trick to capture values per iteration
        def make_evaluator(
            mock_agent: MockQualityAgent, bias_val: float
        ) -> AgentEvaluator:
            def evaluator(fragment: Fragment) -> Tuple[float, Dict[str, Any]]:
                score, meta = mock_agent(fragment)
                score = max(0.0, min(1.0, score + bias_val))
                meta["bias"] = bias_val
                return round(score, 4), meta

            return evaluator

        agents[name] = make_evaluator(mock, bias)

    return agents
