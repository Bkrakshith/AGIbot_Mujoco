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
    # Command curriculum: velocity ranges (vx, vy, yaw) are scaled by this
    # factor in this stage; the height command is not. The FINAL stage must
    # be 1.0 (full envelope, pinned by tests/test_curriculum.py). The vendor
    # X1 training uses a command curriculum for the same reason: on the X1's
    # 6 cm feet, stepping attempts at up to 1.2 m/s from the start end in
    # falls, and PPO settles on standing still (teacher_v1 s1, 2026-09-25).
    cmd_scale: float = 1.0
    # Survival gate: mean eval episode length / episode_length must reach this.
    # Tracking and air time alone let stages pass while the robot still fell
    # in most episodes (teacher_v4 s2: ~50 % falls). 0.0 disables.
    gate_survival: float = 0.0
    # Domain-randomisation curriculum: every DR range is interpolated from its
    # nominal value (k = 0) to the full configs/domain_rand.yaml range (k = 1);
    # latency and obs noise scale the same way. The FINAL stage must be 1.0.
    # Why (ablate_nodr, 2026-09-25): with the full X1 DR from step 0 the X1
    # never formed a gait (air time fell to 0.0017, ep_len ~470/2000); with
    # DR off it stepped (air 0.0067) and survived ~1940/2000.
    dr_scale: float = 1.0


def _load_yaml(name: str) -> DotDict:
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return DotDict.wrap(yaml.safe_load(f))


def stage_commands(commands, stage: "Stage"):
    """The command ranges a stage samples from: velocity ranges scaled by
    stage.cmd_scale, everything else unchanged. Used by the env and by
    curriculum.do_nothing_floor, so gates follow the stage's distribution."""
    k = float(stage.cmd_scale)
    scaled = {r: [k * float(v) for v in commands[r]]
              for r in ("vx_range", "vy_range", "yaw_range")}
    return DotDict.wrap({**commands, **scaled})


# Nominal value each DR range collapses to at dr_scale = 0.
_DR_NOMINAL = {
    ("ground", "friction_range"): 1.0,
    ("ground", "restitution_range"): 0.0,
    ("body", "link_mass_scale_range"): 1.0,
    ("body", "base_mass_delta_range"): 0.0,
    ("body", "base_com_shift_range"): 0.0,
    ("actuation", "pd_gain_scale_range"): 1.0,
    ("actuation", "joint_damping_scale_range"): 1.0,
    ("actuation", "armature_scale_range"): 1.0,
    ("actuation", "motor_strength_range"): 1.0,
}


def stage_domain_rand(dr, stage: "Stage"):
    """The DR a stage samples from: each range pulled toward nominal by
    stage.dr_scale (see _DR_NOMINAL), latency bounds and obs-noise stds
    scaled by it. Payload and push are staged separately."""
    k = float(stage.dr_scale)
    out = {sec: dict(dr[sec]) for sec in ("ground", "body", "actuation", "obs_noise")}
    for (sec, key), nom in _DR_NOMINAL.items():
        out[sec][key] = [nom + k * (float(v) - nom) for v in dr[sec][key]]
    lo, hi = dr.actuation.action_latency_steps
    out["actuation"]["action_latency_steps"] = [int(round(k * lo)), int(round(k * hi))]
    out["obs_noise"] = {n: k * float(v) for n, v in dr.obs_noise.items()}
    return DotDict.wrap({**dr, **out})


def apply_torque_limits(mj_model, actuators) -> None:
    """Optional actuators.yaml `torque_limits` (per actuator, Nm) overriding
    the MJCF ctrlrange in place. Must run BEFORE mjx.put_model: MuJoCo clamps
    ctrl to ctrlrange, so changing only the env's PD clip would not suffice.
    Used identically by the MJX env and the CPU evaluator."""
    lim = actuators.get("torque_limits") if hasattr(actuators, "get") else None
    if lim is None:
        return
    import numpy as np
    lim = np.asarray(lim, dtype=float)
    if lim.shape != (mj_model.nu,):
        raise ValueError(f"torque_limits needs {mj_model.nu} values, got {lim.shape}")
    mj_model.actuator_ctrlrange[:, 0] = -lim
    mj_model.actuator_ctrlrange[:, 1] = lim


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
