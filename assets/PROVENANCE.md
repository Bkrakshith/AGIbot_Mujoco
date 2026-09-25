# Asset provenance

## Source of truth

- **Robot**: AgiBot X1, 36.2 kg, 12 actuated leg joints (waist and arms are
  fixed links in the vendor model).
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

## DoF layout (compiled: nq=19, nv=18, nu=12)

qpos = 7 (free base) + 12 hinges. **qpos order = actuator order**
(pinned by `tests/test_model.py::test_qpos_order_equals_actuator_order`):

| idx | joints |
|---|---|
| 0–5  | left hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll |
| 6–11 | right leg, same order |

The X1 hip-pitch axis is mounted at 45°, and hip pitch/roll/yaw and ankle
roll mirror their sign between legs. Mirror signs are
`[-1, -1, -1, 1, 1, -1]` (`analysis/robot_profile.x1.yaml`, verified).

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
6. `home` keyframe = `configs/actuators.yaml` default_pose (the X1 training
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

## `x1_mjx.xml` (training, primitive contacts)

Vendor collision geoms removed. Primitive set, bitmask 1 floor | 2 feet |
4 hands | 8 torso | 16 foot-self:

- `left/right_foot`: box, half-extents 0.073 × 0.034 × 0.01, fitted by the
  builder to the four vendor sole points plus their radius, oriented
  world-aligned at `home` (local z vertical). Floor only.
- `left/right_foot_self`: sphere r = 0.04, foot–foot only.
- `left/right_hand`: sphere r = 0.04 at the wrist-roll link.
- `torso_capsule`: r = 0.10, vertical at `home`, 0.08 → 0.33 m above the
  torso-link origin.
- `<option iterations="2" ls_iterations="4" solver="Newton" cone="pyramidal"/>`.

## Measured

- Mass 36.16 kg (URDF sum 35.3 kg + 0.1 kg payload placeholders; the
  MJCF inertials differ slightly from the URDF).
- Zero-action PD hold at `home` with the X1 gains stands ~2.2 s, then tips
  (correct: balance is the policy's job).
- Static stance margin at `home`: m̂ = 0.90 (6.6 of 7.3 cm, fore-aft).
