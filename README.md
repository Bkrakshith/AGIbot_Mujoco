# Mujoco_agibot: AgiBot X1 stand + walk with a CPG reference gait

MuJoCo MJX + JAX + brax PPO pipeline that trains the AgiBot X1 to stand and
walk. A Central Pattern Generator supplies the vendor's X1 swing reference and
a gait clock; the policy is a PPO teacher with privileged observations,
distilled into a deployable history-CNN student and exported as ONNX + JSON
sidecar. The robot model and training parameters come from
[agibot_x1_train](https://github.com/AgibotTech/agibot_x1_train), which uses
Isaac Gym; only the files needed here were imported. The pipeline is ported
from the H1-2 stack and follows the `training-humanoid-rl-policies` skill.

## Layout

| path | what |
|---|---|
| `assets/x1_vendor/` | imported subset of agibot_x1_train: `urdf/x1.urdf`, `mjcf/xyber_x1_serial.xml`, 30 meshes, `SOURCE.md` |
| `assets/` | generated `x1_mjx.xml` (training, primitive contacts), `x1_full.xml` (eval, vendor contacts), `PROVENANCE.md` |
| `configs/` | `actuators.yaml` (layout, gains, default pose), `rewards.yaml` (weights, **CPG**, termination), `train.yaml` (commands, PPO, curriculum), `domain_rand.yaml` |
| `src/x1_locomotion/` | `env.py`, `cpg.py`, `rewards.py`, `randomize.py`, `ppo_teacher.py`, `curriculum.py`, `distill_student.py`, `cpu_eval.py`, `layout.py` |
| `train/` | `train_teacher.py` (`--dry_run`), `train_student.py`, `play.py` (video) |
| `eval/` | `run_battery.py` (9 stand/walk scenarios), `metrics.py` |
| `export/` | torch → ONNX + sidecar, verification |
| `analysis/` | X1 robot profile for the skill's static-stability / reach scripts |
| `scripts/` | `setup_env.sh`, `build_assets.py`, `bench.py` |

## Parameters taken from the X1 training config

| what | value |
|---|---|
| PD gains (hip pitch/roll/yaw, knee, ankle pitch/roll) | kp 30/40/35/100/35/35, kd 3/3/4/10/0.5/0.5 |
| default pose | L (0.4, 0.05, −0.31, 0.49, −0.21, 0), R mirrored |
| action scale | 0.5 rad, all joints |
| policy rate | 100 Hz (PD here at 500 Hz) |
| gait cycle / reference | 0.7 s; swing delta L (0.25, 0.05, −0.11, 0.35, −0.16, 0); 0.1 dead band |
| commands | vx [−0.4, 1.2], vy ±0.4, yaw ±0.6 |
| base height | 0.61 (measured 0.613 at the default pose) |
| DR | friction [0.2, 1.3], base mass ±3 kg, CoM ±5 cm, gains/torque ±20 %, latency 0–40 ms, joint damping ×[0.3, 1.5], armature 0.0001–0.05 |
| obs noise | X1 scales × 1.5 |

## Control and observation contract

- Actions (12): leg position targets `q0 + 0.5·a`. PD at 500 Hz, policy at
  100 Hz.
- Student obs (48): projected gravity 3, gyro 3, q−q0 12, qd 12, previous
  action 12, command 4 (vx, vy, yaw rate, pelvis height), gait clock 2
  (sin θ, cos θ, 0.7 s cycle). No base linear velocity.
- Privileged obs (17): payload 8 (always zero here), base lin vel 3,
  friction 1, push 3, kp/kd scale 2.

## CPG

`cpg.cpg_reference(θ, cmd, cfg, q0)`: the stance leg holds q0 and the swing
leg follows q0 + gate·|sin θ|·delta. The left leg swings while sin θ < −0.1
and the right while sin θ > 0.1; in between both feet are down (6 % double
support). `gate` rises to 1 by ‖(vx, vy, ωz)‖ = 0.2, so standing commands
give q0. Rewards use it as `gait_imitation` (joint MSE) and
`gait_contact_swing/stance` (contact timing against the same windows), all
speed-gated to zero when standing.

## Curriculum (configs/train.yaml)

| stage | pushes | cmd switches | budget | gates (tracking / air time) |
|---|---|---|---|---|
| s1_nominal | 0 | no | 60M | 0.879 / 0.003 |
| s2_gait | 0 | no | 40M | 0.879 / 0.003 |
| s3_pushes | 30 N | yes | 60M | 0.897 / 0.0025 |
| s4_robust | 60 N | yes | 60M | 0.897 / 0.0025 |

Tracking gates = (do-nothing floor + 1)/2 with floors 0.757 / 0.794.
~220M steps ≈ 5 h at the measured 12.5k steps/s (16384 envs, RTX 3070).

## Status (2026-09-25)

Assets generated from the imported X1 files, 84 tests pass, GPU bench done,
teacher dry run through the real brax scan passes at 16384 envs, CPU battery
runs on the smoke checkpoint. **No policy trained yet.** Next:
`train_teacher.py --run_name teacher_v1 --backend gpu`.

See `CLAUDE.md` for the rules and the full command list.
