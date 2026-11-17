"""
config.py
=========
Central configuration loader for the LangGraph Nanoindentation multi-agent system.

Loads YAML defaults from ``config/`` with environment-variable overrides.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class ExperimentSettings(BaseModel):
    name: str = "nanoindent_experiment"
    max_iterations: int = 10
    ic_max_cycles: int = 12
    max_hypothesis_revisions: int = 3
    min_hypothesis_critic_passes: int = 2
    hypothesis_refresh_interval: int = 3
    stall_threshold: int = 3
    # Data collection goals
    min_indent_arrays_for_confirmation: int = 2
    min_visited_locations_for_confirmation: int = 3
    min_instrument_actions_before_confirmation: int = 4
    target_indent_count: int = 25
    min_indents_before_conclude: int = 10
    require_multiple_locations_for_confirmation: bool = True


class MemorySettings(BaseModel):
    base_dir: str = "./experiment_memory"
    checkpoint_db: str = "checkpoints.sqlite"
    store_db: str = "episodic_store.sqlite"


class InstrumentSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 5000
    token: str = ""
    timeout: int = 300
    require_approval: bool = True
    sim_mode: bool = False

    @property
    def url(self) -> str:
        env_url = os.environ.get("NANOINDENT_SERVER_URL")
        if env_url:
            return env_url.rstrip("/")
        host = os.environ.get("NANOINDENT_HOST", self.host)
        port = int(os.environ.get("NANOINDENT_PORT", str(self.port)))
        return f"http://{host}:{port}"


class ParamControlSettings(BaseModel):
    """One method parameter's operator/agent contract — see
    ``nanoindentation_agents.method.ParameterSpec``, which this converts to."""
    mode: str = "fixed"           # "fixed" -> use `default` | "agent" -> agent may choose
    default: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    candidates: list[float] = Field(default_factory=list)
    units: str = ""
    agent_allowed: bool = True    # False -> mode="agent" is REFUSED at config load
    note: str = ""                # operator prior knowledge, injected into the LLM prompt


class ChoiceControlSettings(BaseModel):
    """One categorical feature's fixed/agent control contract."""
    mode: str = "fixed"
    default: str = ""
    candidates: list[str] = Field(default_factory=list)
    agent_allowed: bool = True
    note: str = ""


class MethodControlSettings(BaseModel):
    """Bounds + prior knowledge for agent control of indentation method
    parameters. See ``nanoindentation_agents.method.MethodConstraints``."""
    control_mode: str = "depth_controlled"     # "depth_controlled" | "load_controlled"
    control_mode_agent_selectable: bool = False
    derive_cite_depth: bool = True
    cite_depth_frac: float = 0.25
    spacing_depth_multiplier: float = 20.0
    params: dict[str, ParamControlSettings] = Field(default_factory=dict)


class GridControlSettings(BaseModel):
    """Bounds and authority for indentation-array geometry."""
    params: dict[str, ParamControlSettings] = Field(default_factory=lambda: {
        "grid_nx": ParamControlSettings(
            mode="fixed", default=2, min=1, max=5,
            candidates=[1, 2, 3, 4, 5], units="count",
            note="Prefer the smallest grid that can answer the objective."),
        "grid_ny": ParamControlSettings(
            mode="fixed", default=2, min=1, max=5,
            candidates=[1, 2, 3, 4, 5], units="count",
            note="Prefer the smallest grid that can answer the objective."),
        "spacing_um": ParamControlSettings(
            mode="fixed", default=15.0, min=3.0, max=100.0,
            candidates=[6.0, 10.0, 15.0, 20.0, 30.0], units="um",
            note="Keep neighboring plastic zones separated and stay inside the feature."),
        "rotation_deg": ParamControlSettings(
            mode="fixed", default=0.0, min=-180.0, max=180.0,
            candidates=[0.0, 45.0, 90.0], units="deg"),
    })


class MeasurementStrategySettings(BaseModel):
    """Opt-in LLM advisor for per-site measurement recipe selection.

    Disabled by default so existing campaigns remain behavior-identical. When
    enabled, only features whose own mode is ``agent`` may change.
    """
    enabled: bool = False
    use_llm: bool = True
    llm_timeout_s: float = 60.0
    min_confidence: float = 0.55
    require_human_approval_for_agent_changes: bool = True
    site_action: ChoiceControlSettings = Field(default_factory=lambda: ChoiceControlSettings(
        mode="fixed", default="indent", candidates=["indent", "skip"],
        note="Skipping avoids an unnecessary or invalid indentation site."))
    method_type: ChoiceControlSettings = Field(default_factory=lambda: ChoiceControlSettings(
        mode="fixed", default="eh", candidates=["eh"],
        note="Only methods explicitly listed by the operator may be selected."))
    grid_control: GridControlSettings = Field(default_factory=GridControlSettings)


class IndentDefaultSettings(BaseModel):
    method_type: str = "eh"
    grid_nx: int = 2
    grid_ny: int = 2
    spacing_um: float = 15.0
    rotation: float = 0.0
    setup: dict[str, Any] = Field(default_factory=lambda: {
        "poisson": 0.3,
        "metal": 0,
        "drift": 0,
        "target load": 1000,
        "target depth": 1000,
        "surface approach distance": 2000,
        "surface approach velocity": 100,
        "strain rate": 0.2,
    })
    review_changes: dict[str, Any] = Field(default_factory=lambda: {
        "poissons ratio": 0.3,
        "Drift": 0,  # new add July 18,2027
        "Min Depth for Results": 30,
        "Cite at Depth or Load": 0,
        "Stiffness for Contact": 50,
        "Depth to Cite Results (nm)": 250,
        "Metal : Pile up (0) or No Pile Up (1)": 0,
    })
    method_control: MethodControlSettings = Field(default_factory=lambda: MethodControlSettings(
        params={
            "target depth": ParamControlSettings(
                mode="fixed", default=1000, min=150, max=2000,
                candidates=[200, 300, 600, 1000, 1500], units="nm",
                note="Steel combinatorial library; grains ~5-20 um."),
            "target load": ParamControlSettings(
                mode="fixed", default=1000, min=1, max=1000,
                candidates=[200, 500, 1000], units="mN",
                note="Ceiling in depth-controlled mode; inForce1000 full scale is 1000 mN."),
            "surface approach distance": ParamControlSettings(
                mode="fixed", default=2000, min=2000, max=5000, units="nm",
                agent_allowed=False),
            "surface approach velocity": ParamControlSettings(
                mode="fixed", default=100, units="nm/s", agent_allowed=False),
            "strain rate": ParamControlSettings(
                mode="fixed", default=0.2, min=0.01, max=1.0, units="1/s"),
            "poisson": ParamControlSettings(mode="fixed", default=0.3, agent_allowed=False),
            "metal": ParamControlSettings(mode="fixed", default=0, agent_allowed=False),
            "drift": ParamControlSettings(mode="fixed", default=0, agent_allowed=False),
        }
    ))


class GraphSettings(BaseModel):
    profile: str = "full"
    interrupt_nodes: list[str] = Field(default_factory=lambda: ["user_approval"])
    auto_approve_instruments: bool = False


class LiteratureSettings(BaseModel):
    backend: str = "local"  # "local" or "manager" for online search
    local_dir: str = "./literature"
    max_chunks: int = 8
    use_manager: bool = True  # Enable Crossref, arXiv, ChemRxiv search
    manager_max_papers: int = 3
    manager_max_candidates: int = 20
    year_min: int = 2018
    year_max: int = 2026
    fallback_to_local_files: bool = True
    rss_feeds: list[str] = Field(default_factory=list)  # Optional custom RSS feeds
    rss_min_term_matches: Optional[int] = 2  # Min keyword matches for RSS filtering


class RoutingSettings(BaseModel):
    force_hypothesis_approval_after_revisions: bool = True


class ModelRoleSettings(BaseModel):
    model: str = "Qwen/Qwen2.5-72B-Instruct"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    vision: bool = False
    provider: str = "huggingface"
    local_path: Optional[str] = None
    fallback_provider: Optional[str] = "huggingface"
    fallback_model: Optional[str] = None
    inference_provider: Optional[str] = None


class ModelsSettings(BaseModel):
    default_max_tokens: int = 2048
    default_temperature: float = 0.7
    roles: dict[str, ModelRoleSettings] = Field(default_factory=dict)

    def role(self, name: str) -> ModelRoleSettings:
        if name in self.roles:
            return self.roles[name]
        return ModelRoleSettings(
            model=os.environ.get("NANOINDENT_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
            temperature=self.default_temperature,
            max_tokens=self.default_max_tokens,
        )


class PromptRoleMapping(BaseModel):
    system: Optional[str] = None
    user: Optional[str] = None
    system_key: Optional[str] = None


class PromptsSettings(BaseModel):
    roles: dict[str, PromptRoleMapping] = Field(default_factory=dict)


class LangSmithSettings(BaseModel):
    enabled: bool = False
    project: str = "nanoindent-agent"
    tracing_v2: bool = True


class AppConfig(BaseModel):
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    instrument: InstrumentSettings = Field(default_factory=InstrumentSettings)
    indent_defaults: IndentDefaultSettings = Field(default_factory=IndentDefaultSettings)
    measurement_strategy: MeasurementStrategySettings = Field(
        default_factory=MeasurementStrategySettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    literature: LiteratureSettings = Field(default_factory=LiteratureSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    prompts: PromptsSettings = Field(default_factory=PromptsSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    research_question: str = ""
    experiment_context: str = ""

    @property
    def memory_dir(self) -> Path:
        return Path(self.memory.base_dir).resolve()

    @property
    def checkpoint_path(self) -> Path:
        return self.memory_dir / self.memory.checkpoint_db

    @property
    def store_path(self) -> Path:
        return self.memory_dir / self.memory.store_db


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NANOINDENT_", extra="ignore")

    config_path: Optional[str] = None
    memory_dir: Optional[str] = None
    server_url: Optional[str] = None


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """
    Load merged configuration from default YAML files and optional experiment overlay.
    """
    merged: dict[str, Any] = {}
    for name in ("default.yaml", "models.yaml", "prompts.yaml"):
        merged = _deep_merge(merged, _load_yaml(_CONFIG_DIR / name))

    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            candidates = [
                path,
                _REPO_ROOT / path,
                _CONFIG_DIR / "experiments" / path.name,
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                overlay = json.load(f)
        else:
            overlay = _load_yaml(path)
        merged = _deep_merge(merged, overlay)

    env = EnvSettings()
    if env.memory_dir:
        merged.setdefault("memory", {})["base_dir"] = env.memory_dir

    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes"):
        merged.setdefault("langsmith", {})["enabled"] = True

    cfg = AppConfig.model_validate(merged)
    return cfg


_active_override: Optional[AppConfig] = None


@lru_cache(maxsize=4)
def get_config(config_path: Optional[str] = None) -> AppConfig:
    if _active_override is not None and config_path is None:
        return _active_override
    return load_config(config_path)


def set_active_config(config: AppConfig) -> None:
    """Pin config for the current process."""
    global _active_override
    _active_override = config
    get_config.cache_clear()


def reset_config_cache() -> None:
    global _active_override
    _active_override = None
    get_config.cache_clear()
