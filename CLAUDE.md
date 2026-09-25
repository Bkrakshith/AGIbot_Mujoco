# AgiBot X1 stand + walk (CPG-guided) on MJX — project rules

AgiBot X1 (vendor MJCF from agibot_x1_train, 12 leg DoF, 36 kg). PPO teacher
with privileged obs, trained through a gated curriculum under per-episode
domain randomisation. A Central Pattern Generator reference gait
(`src/x1_locomotion/cpg.py`, the vendor X1 swing reference) is exposed to the
policy as a gait clock. The teacher is then distilled into a history-CNN
student and exported as ONNX + JSON sidecar.

The pipeline is ported from the validated H1-2 stack at
`/home/rakshith/UnitreeH1_2_Factory/Mujoco_unitree`. X1 physical and training
parameters come from agibot_x1_train `x1_dh_stand_config.py` and are cited
inline as "X1 cfg" in `configs/*.yaml`. Method and failure catalogue: the
`training-humanoid-rl-policies` skill.

## Hard rules

- All physical ranges, reward weights, curriculum gates live in `configs/*.yaml`.
- All DoF sizes/indices come from `src/x1_locomotion/layout.py`
  (`configs/actuators.yaml` `layout:`). Never hardcode 12 / 48.
- Assets are GENERATED from `assets/x1_vendor/`: edit
  `scripts/build_assets.py`, re-run it, run `pytest tests/test_model.py`.
  Import nothing else from the vendor repo unless it is used.
- Tracking gates = (do-nothing floor + 1) / 2, pinned by
  `tests/test_curriculum.py`. If you change commands or `tracking_sigma`,
  recompute the gates (`curriculum.do_nothing_floor`).
- A stalled stage: more budget first, then reward/curriculum. Never shrink
  the task (speeds, pushes) to pass a gate. One change per run.
- Select the best checkpoint per stage, not the last.
- The battery (`eval/run_battery.py`, full model, >= 50 eps, Wilson CIs) is
  the only number that counts. Verify the instrument before believing a
  regression.
- torch only under `export/`. No checkpoints/ONNX/videos in git.

## Machine quirks (this Linux box)

- ALWAYS prefix long / multi-process commands with `taskset -c 0-7,10-31`
  (logical CPUs 8/9 segfault under load; unpinned pytest segfaults).
- GPU: `--backend gpu` (RTX 3070 8 GB). `backend.py` preloads the venv's
  NVIDIA libs to beat a stale libnvJitLink on LD_LIBRARY_PATH.
- `pip` pinned at 26.2 in `.venv`: 26.2.1 fails to unzip wheels on the
  py311tmp Python.

## Workflow

```bash
source .venv/bin/activate          # created by scripts/setup_env.sh
P="taskset -c 0-7,10-31"
python scripts/build_assets.py                                  # regenerate MJCFs
$P python -m pytest tests/ -q                                   # 84 tests
$P python scripts/bench.py --backend gpu --num_envs 8192,16384  # throughput
$P python train/train_teacher.py --run_name smoke --dry_run --backend gpu
$P python train/train_teacher.py --run_name teacher_v1 --backend gpu [--resume]
$P python eval/run_battery.py --checkpoint outputs/checkpoints/teacher_v1/ckpt_N
$P python train/play.py --checkpoint ... --scenario c_walk_forward   # video
$P python train/train_student.py --teacher_ckpt ... --run_name student_v1 --backend gpu
python export/params_to_torch.py --checkpoint .../student_v1/ckpt_final
python export/export_onnx.py --torch outputs/onnx/student.pt
python export/verify_onnx.py --onnx outputs/onnx/student.onnx --torch outputs/onnx/student.pt
$P python eval/run_battery.py --onnx outputs/onnx/student.onnx --gate 0.9
```

## Known quirks

- MJX logs `Accessing contact directly from Data is deprecated` and XLA logs
  `xtile_compiler` / `Delay kernel timed out` lines on GPU: harmless.
- Zero-action PD hold stands ~2.2 s then tips: correct physics.
- Vendor joint ranges are ±3.14 rad everywhere (no real limits exist in
  either vendor file); action authority (±0.5 rad) is the effective limit.
- Ankle torque limit 18 Nm (vendor MJCF) vs 80 Nm (vendor URDF); we use 18.
- Not ported from the X1 training: terrain (flat only), motor-offset DR,
  per-joint friction DR, and its command curriculum.
