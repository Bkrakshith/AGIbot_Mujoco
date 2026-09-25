"""Config loading. All physical ranges, reward weights, and curriculum
thresholds live in configs/*.yaml — this module only parses them."""

import dataclasses
import os
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
MJX_XML = os.path.join(ASSETS_DIR, "x1_mjx.xml")
FULL_XML = os.path.join(ASSETS_DIR, "x1_full.xml")


class DotDict(dict):
    """dict with attribute access, recursively applied."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj


@dataclasses.dataclass(frozen=True)
class Stage:
    """One curriculum stage (spec section 7 Phase A). Frozen: a stage change
    means a new env instance and a fresh jit."""

    name: str
    num_timesteps: int
    payload_max_total: float
    payload_symmetric: bool
    push_max_n: float
    arm_range_scale: float
    swap_enabled: bool
    cmd_switch_enabled: bool
    gate_tracking: float
    # Gait-activity gate: mean per-step RAW feet_air_time reward from eval.
    # The tracking kernel alone is gameable by lean-drift (teacher_v3 s1
    # scored 0.78 while never lifting a foot on the full model); air time
    # cannot be earned without actual stepping. 0.0 disables the criterion.
    gate_air_time: float = 0.0


def _load_yaml(name: str) -> DotDict:
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return DotDict.wrap(yaml.safe_load(f))


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively overlay `override` on `base`; scalars and lists replace.

    Task overlays carry only their deltas, so the base
    files stay the single source of truth for everything they do not mention.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _deep_merge(base[k], v) if k in base else v
        return DotDict.wrap(out)
    return override


@dataclasses.dataclass
class Config:
    actuators: DotDict
    rewards: DotDict
    domain_rand: DotDict
    train: DotDict

    @property
    def stages(self) -> list[Stage]:
        return [Stage(**s) for s in self.train.curriculum]


def load_config(overlay: str | None = None) -> Config:
    """Base config, optionally with a task overlay merged over it.

    `overlay` names a file in configs/ holding only the deltas for one task
    (e.g. a future task overlay). Its top-level keys must match the Config fields.
    """
    cfg = Config(
        actuators=_load_yaml("actuators.yaml"),
        rewards=_load_yaml("rewards.yaml"),
        domain_rand=_load_yaml("domain_rand.yaml"),
        train=_load_yaml("train.yaml"),
    )
    if overlay is None:
        return cfg

    extra = _load_yaml(overlay)
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(extra) - known
    if unknown:
        raise ValueError(
            f"{overlay}: unknown top-level section(s) {sorted(unknown)}; "
            f"expected any of {sorted(known)}")
    for section, value in extra.items():
        setattr(cfg, section, _deep_merge(getattr(cfg, section), value))
    return cfg
