# Asset provenance

## Source of truth

- **Robot**: AgiBot X1, 36.2 kg. 12 leg joints (vendor) + 12 arm joints
  re-enabled from the vendor URDF axes (task-commanded, not policy-driven);
  the waist stays fixed.
- **Source repo**: https://github.com/AgibotTech/agibot_x1_train
  (`resources/robots/x1/`). Only a subset is imported, see
  `x1_vendor/SOURCE.md`:
  - `x1_vendor/mjcf/xyber_x1_serial.xml`: the vendor MuJoCo model (used by
    their sim2sim). **The generator's input.**
  - `x1_vendor/urdf/x1.urdf`: reference only (masses, axes, efforts). Its
    24 linkage meshes were not imported, so it does not render fully.
  - `x1_vendor/meshes/`: the 30 STL files the MJCF references (37 MB).
- **Both XMLs are generated** by `scripts/build_assets.py`. Do not hand-edit;
  change the script or `configs/actuators.yaml` and re-run it, then
  `pytest tests/test_model.py`.

## DoF layout (compiled: nq=31, nv=30, nu=24)

qpos = 7 (free base) + 24 hinges. **qpos order = actuator order**
(pinned by `tests/test_model.py::test_qpos_order_equals_actuator_order`):

| idx | joints | driven by |
|---|---|---|
| 0–5   | left hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll | policy |
| 6–11  | right leg, same order | policy |
| 12–17 | left shoulder_pitch/roll/yaw, elbow_pitch/yaw, wrist_pitch | task arm command (PD) |
| 18–23 | right arm, same order | task arm command (PD) |

The X1 hip-pitch axis is mounted at 45°, and hip pitch/roll/yaw and ankle
roll mirror their sign between legs. Mirror signs are
`[-1, -1, -1, 1, 1, -1]` (`analysis/robot_profile.x1.yaml`, verified).
Arm conventions are identical for both arms: shoulder_pitch −1.57 = forward,
+0.6 = back, ≈ −3.0 = overhead; shoulder_roll −1.57 = out to the side;
elbow_pitch + = flex (pinned by `test_arm_pose_semantics`).

## Changes from the vendor MJCF (both variants)

1. Bodies renamed for the env contract: `x1-body` → `pelvis`,
   `body_pitch` → `torso_link`. Joint names unchanged.
2. Sensors and the vendor `home_default` keyframe stripped.
3. timestep 0.001 → 0.002 (500 Hz PD; the policy runs at 100 Hz as on the X1).
4. **Joint armature 0 → 0.01.** At 500 Hz a zero-armature PD hold spiked to
   10 rad/s joint speed; 0.01 matches the 1 kHz behaviour and lies inside the
   X1 training's armature randomisation [0.0001, 0.05]. Vendor damping (1.0)
   kept.
5. Floor + light; `left/right_payload` bodies (0.05 kg) on the wrist-roll
   links as the hand point (no payload is sampled).
6. **Arm joints added** (`scripts/build_assets.py` ARM_JOINTS): six per arm
   with the axes recorded in the vendor URDF (its arm joints are `fixed` but
   keep their axes; the leg axes match between URDF and MJCF, so the frames
   agree). Ranges, torque limits (40/40/20/20/10/6 Nm) and gains are NOT
   vendor values — chosen to cover the task poses without the arm passing
   through the torso. The torso subtree is moved after the legs so the
   legs stay first in qpos.
7. `home` keyframe = `configs/actuators.yaml` default_pose (the X1 training
   config's `default_joint_angles`), pelvis at **0.613 m** (X1 cfg
   `base_height_target` 0.61), measured so the soles touch the floor.

Vendor quirks kept as-is: every joint range is ±3.14 rad (the URDF too; there
are no real limits to import), and the actuator ctrlrange is hip/knee 150,
hip roll/yaw 50, **ankle 18 Nm**. The URDF says ankle effort 80 Nm; the MJCF
value is the one the vendor's MuJoCo sim2sim runs under, and the more
conservative one.

## `x1_full.xml` (evaluation)

Vendor collision set: base and torso meshes plus four r = 2 mm sole spheres
per foot (sole 0.14 × 0.06 m). Floor conaffinity 7 as in the vendor scene.

## `x1_mjx.xml` (training)

Mesh collision geoms removed. **The foot contact is the vendor's own four
r = 2 mm sole spheres per foot — identical to `x1_full.xml`** (a box sole
here gave backward-walk survival 83 % in training vs 0 % on the eval model;
`tests/test_model.py::test_training_and_eval_soles_are_identical`).
Bitmask 1 floor | 2 feet | 4 hands | 8 torso | 16 foot-self:

- `left/right_sole_0..3`: the vendor sole spheres, floor only.
- `left/right_foot`: NON-colliding reference box over the sole (pose and
  support polygon for the env; half-extents 0.073 × 0.034).
- `left/right_foot_self`: sphere r = 0.04, foot–foot only (self-collision
  termination).
- `left/right_hand`: sphere r = 0.04, floor only (arms are task-commanded;
  an arm–torso touch is not the leg policy's fault).
- `torso_capsule`: inert (conaffinity 0).
- `<option iterations="2" ls_iterations="4" solver="Newton" cone="pyramidal"/>`.

## Measured

- Mass 36.16 kg (URDF sum 35.3 kg + 0.1 kg payload placeholders; the
  MJCF inertials differ slightly from the URDF).
- Every library arm pose is statically feasible: CoM margin ≥ 2.6 cm (both
  arms straight forward) of 7.3 cm.
- `scripts/preflight_model_check.py` (skill): 0 FAIL; WARN only for the
  vendor's ±π leg ranges.
- Zero-action PD hold at `home` with the X1 gains stands ~2.2 s, then tips
  (correct: balance is the policy's job).
- Static stance margin at `home`: m̂ = 0.90 (6.6 of 7.3 cm, fore-aft).
