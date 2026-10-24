"""Load public settings and private keys."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    root: Path
    project: str
    dataset: str
    jev_model: str
    luna_model: str
    luna_reasoning_effort: str
    luna_max_completion_tokens: int
    judge_model: str
    reflection_model: str
    jev_prompt: Path
    luna_prompt: Path
    max_metric_calls: int
    min_reviews: int
    min_rooms: int
    min_judge_agreement: float
    braintrust_key: str
    typesafe_key: str
    openai_key: str


def load_settings(config_path: Path | None = None) -> Settings:
    path = (
        config_path or Path(__file__).resolve().parents[2] / "config.toml"
    ).resolve()
    root = path.parent
    load_dotenv(root / ".env", override=False)
    data = tomllib.loads(path.read_text())

    def setting(section: str, key: str, env: str) -> str:
        return os.getenv(env) or str(data[section][key])

    def prompt_path(key: str, env: str) -> Path:
        value = Path(setting("prompts", key, env))
        return value if value.is_absolute() else root / value

    luna_max_completion_tokens = int(
        setting("models", "luna_max_completion_tokens", "LUNA_MAX_COMPLETION_TOKENS")
    )
    if luna_max_completion_tokens < 1:
        raise ValueError("Luna reply token cap must be positive")

    return Settings(
        root=root,
        project=setting("braintrust", "project", "BRAINTRUST_PROJECT"),
        dataset=setting("braintrust", "dataset", "BRAINTRUST_DATASET"),
        jev_model=setting("models", "jev", "JEV_MODEL"),
        luna_model=setting("models", "luna", "LUNA_MODEL"),
        luna_reasoning_effort=setting(
            "models", "luna_reasoning_effort", "LUNA_REASONING_EFFORT"
        ),
        luna_max_completion_tokens=luna_max_completion_tokens,
        judge_model=setting("models", "judge", "JUDGE_MODEL"),
        reflection_model=setting("models", "reflection", "REFLECTION_MODEL"),
        jev_prompt=prompt_path("jev", "JEV_PROMPT"),
        luna_prompt=prompt_path("luna", "LUNA_PROMPT"),
        max_metric_calls=int(
            os.getenv("MAX_METRIC_CALLS") or data["gepa"]["max_metric_calls"]
        ),
        min_reviews=int(data["gepa"]["min_reviews"]),
        min_rooms=int(data["gepa"]["min_rooms"]),
        min_judge_agreement=float(data["gepa"]["min_judge_agreement"]),
        braintrust_key=os.getenv("BRAINTRUST_API_KEY", ""),
        typesafe_key=os.getenv("TYPESAFE_API_KEY", ""),
        openai_key=os.getenv("OPENAI_API_KEY", ""),
    )
