from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openai_key import resolve_openai_api_key


class ContextCompactor:
    """Builds bounded diagnosis context without modifying raw history."""

    def __init__(self, config: dict[str, Any], history_dir: Path):
        self.config = config
        settings = config.get("context_compaction", {})
        self.enabled = bool(settings.get("enabled", True))
        self.model = str(settings.get("model") or config.get("openai_model", "gpt-5-mini"))
        self.chunk_max_chars = max(10_000, int(settings.get("chunk_max_chars", 60_000)))
        self.chunk_summary_max_chars = max(500, int(settings.get("chunk_summary_max_chars", 2_500)))
        self.iteration_summary_max_chars = max(1_000, int(settings.get("iteration_summary_max_chars", 6_000)))
        self.rolling_summary_max_chars = max(2_000, int(settings.get("rolling_summary_max_chars", 15_000)))
        self.exact_facts_max_chars = max(1_000, int(settings.get("exact_facts_max_chars", 5_000)))
        self.context_dir = history_dir / "context_summaries"
        self.cache_dir = self.context_dir / "cache"
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_path = self.context_dir / "rolling_summary.json"

    def build(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not history:
            return []
        if not self.enabled:
            return [{"history": history}]

        summaries: list[dict[str, Any]] = []
        hashes: list[str] = []
        for entry in history:
            source_hash = self._entry_hash(entry)
            hashes.append(source_hash)
            summaries.append(self._entry_summary(entry, source_hash))

        state = self._load_rolling_state()
        prefix = 0
        if state and state.get("source_hashes"):
            previous_hashes = state["source_hashes"]
            prefix = self._common_prefix(previous_hashes, hashes)
            if prefix == len(hashes) and state.get("summary"):
                rolling = str(state["summary"])
            else:
                rolling = str(state.get("summary", "")) if prefix == len(previous_hashes) else ""
        else:
            rolling = ""

        for index in range(prefix, len(summaries)):
            rolling = self._merge_summary(rolling, summaries[index]["summary"])

        state = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source_hashes": hashes,
            "summary": self._limit_text(rolling, self.rolling_summary_max_chars),
        }
        self.rolling_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        facts = self._exact_facts(history)
        return [
            {"rolling_summary": state["summary"]},
            {"exact_facts": facts},
        ]

    def _entry_summary(self, entry: dict[str, Any], source_hash: str) -> dict[str, Any]:
        cache_path = self.cache_dir / f"{source_hash}.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("summary"):
                    return cached
            except (OSError, ValueError, TypeError):
                pass

        chunks = list(self._entry_chunks(entry))
        chunk_summaries = [self._summarize(chunk, self.chunk_summary_max_chars) for chunk in chunks]
        combined = "\n\n".join(chunk_summaries)
        summary = self._summarize(
            "Combine these complete history-section summaries. Preserve exact candidate IDs, "
            "iteration numbers, metrics, failures, mechanisms, and literature sources.\n\n"
            + combined,
            self.iteration_summary_max_chars,
        )
        result = {
            "source_hash": source_hash,
            "iteration": entry.get("iteration"),
            "summary": self._limit_text(summary, self.iteration_summary_max_chars),
        }
        cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _entry_chunks(self, entry: dict[str, Any]) -> list[str]:
        chunks: list[str] = []
        for key, value in entry.items():
            encoded = json.dumps({key: value}, ensure_ascii=False, indent=2)
            if len(encoded) <= self.chunk_max_chars:
                chunks.append(encoded)
                continue
            if isinstance(value, list):
                batch: list[Any] = []
                for item in value:
                    candidate = json.dumps({key: batch + [item]}, ensure_ascii=False)
                    if batch and len(candidate) > self.chunk_max_chars:
                        chunks.append(json.dumps({key: batch}, ensure_ascii=False, indent=2))
                        batch = []
                    batch.append(item)
                if batch:
                    chunks.append(json.dumps({key: batch}, ensure_ascii=False, indent=2))
                continue
            text = encoded
            for start in range(0, len(text), self.chunk_max_chars):
                chunks.append(text[start : start + self.chunk_max_chars])
        return chunks or ["{}"]

    def _summarize(self, source: str, max_chars: int) -> str:
        prompt = (
            "Summarize the following SFM refinement context for a later diagnosis model. "
            "Preserve exact IDs, numbers, metric direction, failures, mechanisms, and source names. "
            f"Return concise plain text no longer than {max_chars} characters. "
            "Do not invent facts.\n\nSOURCE:\n" + source
        )
        try:
            from openai import OpenAI  # type: ignore

            response = OpenAI(api_key=resolve_openai_api_key()).responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max(256, max_chars // 3),
                reasoning={"effort": "medium"},
            )
            text = response.output_text or ""
            if text:
                return self._limit_text(text, max_chars)
        except Exception:
            pass
        return self._limit_text(source, max_chars)

    def _merge_summary(self, previous: str, current: str) -> str:
        if not previous:
            return self._limit_text(current, self.rolling_summary_max_chars)
        return self._summarize(
            "PREVIOUS ROLLING SUMMARY:\n"
            + previous
            + "\n\nNEW ITERATION SUMMARY:\n"
            + current,
            self.rolling_summary_max_chars,
        )

    def _exact_facts(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for entry in history:
            item: dict[str, Any] = {
                key: entry.get(key)
                for key in ("iteration", "decision", "notes")
                if entry.get(key) is not None
            }
            candidates = []
            for candidate in entry.get("children_created", []) or []:
                if not isinstance(candidate, dict):
                    continue
                candidates.append(
                    {
                        key: candidate.get(key)
                        for key in ("candidate_id", "parent_id", "status", "metrics", "pool_topsis_score")
                        if candidate.get(key) is not None
                    }
                )
            if candidates:
                item["children"] = candidates
            facts.append(item)
        encoded = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= self.exact_facts_max_chars:
            return facts
        # Preserve the newest exact facts first when the ledger exceeds its budget.
        kept: list[dict[str, Any]] = []
        for item in reversed(facts):
            candidate = [item] + kept
            if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) > self.exact_facts_max_chars:
                break
            kept = candidate
        return kept

    @staticmethod
    def _entry_hash(entry: dict[str, Any]) -> str:
        encoded = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _common_prefix(left: list[str], right: list[str]) -> int:
        count = 0
        for old, new in zip(left, right):
            if old != new:
                break
            count += 1
        return count

    @staticmethod
    def _load_rolling_state_from(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _load_rolling_state(self) -> dict[str, Any] | None:
        return self._load_rolling_state_from(self.rolling_path)

    @staticmethod
    def _limit_text(value: Any, max_chars: int) -> str:
        text = str(value or "")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 48].rstrip() + "...[summary truncated]"
