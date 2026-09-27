# Mujoco_agibot: AgiBot X1 robust stand + walk (CPG-guided) with task arm poses

> **Status: under active development — tested in simulation only.**
> The policies in this repository have been trained and evaluated in MuJoCo
> only. Sim-to-real transfer is **pending**: nothing here has been run on a
> physical robot yet, and results will be updated here once hardware testing
> has been done.
>
> **Disclaimer.** This software is provided "as is", without warranty of any
> kind. The author is not responsible for any sim-to-real deployment of these
> policies or of any part of this repository. Running them on real hardware
> is done entirely at the user's own risk, including any damage to the robot,
> its surroundings or people. Always follow the robot manufacturer's safety
> procedures (supervision, emergency stop, safety tether) when testing
> learned controllers.

MuJoCo MJX + JAX + brax PPO pipeline that trains the AgiBot X1 to stand and
walk robustly under the **full vendor domain randomisation and sensor noise**,
while its arms move through task poses (forward, sideways, overhead, back,
carry). A privileged PPO teacher is trained through a 7-stage curriculum,
distilled into a deployable history-based student, and exported as ONNX +
JSON sidecar. Robot model and training parameters come from
[agibot_x1_train](https://github.com/AgibotTech/agibot_x1_train) (only the
files needed were imported).

## Simulation results (2026-09-27)

Exported policy (simulation-validated only): `outputs/onnx/student.onnx` (+ `student.json` sidecar),
from `student_v3` ← teacher `teacher_v13`. The ONNX file itself, evaluated on the **full-fidelity
model** with **full X1 sensor noise**, 50 episodes per scenario:

| scenario | survival |
|---|---|
| stand still | 100 % |
| stand + 40 N pushes | 96 % |
| walk forward 0.8 m/s | 92 % |
| walk backward, rotate in place, sidestep, turn while walking | 100 % |
| walk + 35 N pushes | 96 % |
| walk with arms forward / sideways / overhead | 100 % |
| walk with arms moving through poses | 98 % |
| stop-and-go | 100 % |

Torso while walking (teacher, full model): mean pitch within 2°, pitch
peak-to-peak ≤ 1.5°, roll peak-to-peak 3–5°; commanded speed tracked
(0.6 → 0.56 m/s). Results JSON: `outputs/eval/`. Videos: `outputs/videos/final/`.

## Layout

| path | what |
|---|---|
| `assets/x1_vendor/` | imported subset of agibot_x1_train (URDF, MJCF, 30 meshes, `SOURCE.md`) |
| `assets/` | generated `x1_mjx.xml` (training) / `x1_full.xml` (eval) — same joints AND same vendor sole contacts; `PROVENANCE.md` |
| `configs/` | `actuators.yaml` (layout, gains, pose), `rewards.yaml` (weights, CPG, stability terms), `train.yaml` (commands, PPO, 7-stage curriculum, distill), `domain_rand.yaml` (X1 DR, noise, arm poses), `student_push_mix.yaml`, `ablations/` |
| `src/x1_locomotion/` | `env.py`, `cpg.py`, `rewards.py`, `randomize.py`, `ppo_teacher.py`, `curriculum.py`, `distill_student.py`, `cpu_eval.py`, `layout.py`, `config.py` |
| `train/` | `train_teacher.py`, `train_student.py`, `play.py` (videos), `view.py` (interactive) |
| `eval/` | `run_battery.py` (13 scenarios, `--obs_noise`, `--onnx`, `--gate`) |
| `export/` | JAX → torch → ONNX + sidecar, parity checks |
| `scripts/` | `setup_env.sh`, `build_assets.py`, `bench.py` |

## Control and observation contract (sidecar is authoritative)

- Actions (12): leg position targets `q0 + 0.5·a`; PD 500 Hz in sim, policy 100 Hz.
- Arms (12 DoF) follow a **task arm-pose command** with their own PD; the
  policy observes them but does not actuate them.
- Student obs (84) per step: projected gravity 3, gyro 3, q−q0 24, qd 24,
  previous action 12, command 4 (vx, vy, yaw rate, pelvis height), arm-pose
  command 12, gait clock 2 (0.7 s cycle). Policy input: last 50 steps,
  raw values (normalisation inside the graph; prime the buffer with the
  published mean on reset).

## Key design decisions (why it works)

- **Curriculum ramps one difficulty per stage**: commands (0.4→1.0),
  arm poses (0→0.5→1.0), DR (0→0.5→0.75→1.0), pushes (30→60 N); gates:
  tracking (do-nothing floor rule), air time, survival (final ≥ 0.8).
- **Teacher privileged obs = every randomised latent that is inferable from
  history** (friction, gains, motor strength, latency, mass/CoM) — NOT the
  instantaneous push force.
- **Teacher trains noise-free; the student gets the full X1 noise** and
  trains its own trunk + history CNN (DAgger, labels from clean obs).
- Train and eval models share the vendor 4-point sole.

## Workflow

```bash
source .venv/bin/activate; P="taskset -c 0-7,10-31"
python scripts/build_assets.py
$P python -m pytest tests/ -q
$P python train/train_teacher.py --run_name teacher_vN --backend gpu
$P python train/train_student.py --teacher_ckpt outputs/checkpoints/<teacher>/ckpt_N --run_name student_vN --backend gpu
$P python eval/run_battery.py --checkpoint outputs/checkpoints/student_vN/ckpt_best --obs_noise
python export/params_to_torch.py --checkpoint .../ckpt_best --out outputs/onnx/student.pt
python export/export_onnx.py --torch outputs/onnx/student.pt
python export/verify_onnx.py --onnx outputs/onnx/student.onnx --torch outputs/onnx/student.pt
$P python eval/run_battery.py --onnx outputs/onnx/student.onnx --obs_noise --gate 0.9
$P python train/view.py --onnx outputs/onnx/student.onnx --obs_noise   # interactive
```

## Sim-to-real status: pending

Not tested on hardware. Not yet modelled in simulation (all present in the
vendor X1 training config and planned before any hardware test):

per-joint gain/strength spread, encoder zero offsets, joint Coulomb
friction, and sensor (observation) latency. The torque/joint limits and gains
of the arms are assumed, not vendor values. Sim-to-real results will be added
to this README once testing has been done.
