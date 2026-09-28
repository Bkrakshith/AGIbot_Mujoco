# X1 two-arm box squeeze: MuJoCo feasibility

Box 0.51 w x 0.52 d x 0.52 h m, 2.0 kg. Pelvis fixed at 0.613 m (no squat).
Arm PD at the sidecar gains; arm ranges, gains and torque limits are assumed
values (scripts/build_assets.py ARM_JOINTS), not vendor data.

Gripper: a closed OmniPicker is modelled as a rigid pad 80 x 120 x 60 mm on
the wrist_roll/payload frame. It runs from 0.03 to 0.15 m along -y and is
centred on the wrist axis. It adds 0.43 kg per wrist, so the robot model is
37.02 kg. The vendor wrist_roll STL already reaches 0.104 m past the wrist,
so the pad overlaps it and does not add to the length. The vendor rates each
arm for 0.5 kg; this test puts about 1 kg per arm.

Scripts are in `manip/`: build_scene, ik, hold, balance, place, render. Raw
numbers are in `manip/out/*.json`. The video is
`outputs/videos/manip/squeeze_pick_carry_place.mp4`.

## 1. Geometry: what can touch the box

- The forearm is a Ø112 mm cylinder. The pad's medial face sits only 30 mm
  from the wrist axis.
- The wrist-pitch axis is parallel to the pad normal. The pad face is
  therefore always parallel to the forearm axis, and the forearm stands
  about 25 mm proud of the pad.
- A flat-palm squeeze needs the hand angled 10-20 deg inward, so that the
  forearm clears the face.
- The arms are short. The shoulder is at x ≈ 0 and shoulder-to-wrist is
  0.42 m. The box CoM is at x ≥ 0.36 whenever the box touches the chest
  (the chest front is at x = 0.099).

IK sweep (damped least squares, both arms, box centred on y = 0, contact at
mid-height, the contact element 4 cm inside the face edges, other arm
proxies 1 cm clear of the box and 5 mm clear of the torso):

| contact | box-bottom band | box x range | contact x at chest (box x 0.36-0.37) |
|---|---|---|---|
| pad face (≤15 deg to face) | 0.40-1.15 m | 0.36-0.72 | at the CoM (0.36) only for bottom 0.65-0.80; 0.14-0.30 elsewhere |
| forearm line | 0.50-1.00 m | 0.36-0.54 | 0.14-0.26 (10-22 cm behind the CoM) |
| pad tip (V hands) | 0.40-1.10 m | 0.36-0.76 | 0.35-0.36 |

- Below a box bottom of 0.40 m, the box hits the legs/pelvis at box x < 0.38.
- The chest-hold is the box touching the chest, at box x 0.37 and box
  bottom 0.75 m.

Contact pose, pads on the faces at the CoM (d = 0), rad, per arm
[sh_pitch, sh_roll, sh_yaw, el_pitch, el_yaw, wr_pitch]:

- L: -0.744 -1.082 -0.780 1.345 1.107 -0.765
- R: -0.756 -1.054 -0.765 1.327 1.101 -0.688

Wrist pitch sits within 0.04 rad of its ±0.8 range limit.

Chest-hold command (d = 0.14, box raised 8 cm in command):

- L: -0.764 0.141 -0.047 1.335 -0.132 -0.071
- R: -0.779 0.052 -0.105 1.327 -0.034 -0.076

This pose is outside the policy's training arm library (nearest: "forward
reach").

## 2. Static hold (pelvis fixed, 5 s)

**Full collision set (forearm and upper-arm capsules collide): the box was
never held.**

- d ≤ 0.08: the forearms reach the face first, 10-20 cm behind the CoM. The
  box pivots about the grip line (about y) and falls once the table is
  removed.
- d ≥ 0.10: the approach sweeps the forearm into the box and knocks it off
  the table.
- Torsional friction on the contacts (condim 4, 0.02 m) did not change this.

**Pads only (the forearm kept off the box), grip at the CoM x:**

| mu | d (m) | N per side (N) | held | drop 1-5 s (mm) | tilt (deg) |
|---|---|---|---|---|---|
| 0.3 | 0.12 / 0.14 / 0.16 | 15 / 37 / 41 | no / no / no (11.9 mm) | 735 / 69 / 12 | 1 / 26 / 6 |
| 0.5 | 0.10 / 0.12 / 0.14 / 0.16 | 33 / 32 / 38 / 41 | no / yes / yes / yes | 19 / 8 / 8 / 7 | 23 / 8 / 11 / 1 |
| 0.8 | 0.08 / 0.10 / 0.12 / 0.16 | 26 / 33 / 32 / 41 | no / no / yes / yes | 19 / 20 / 9 / 8 | 19 / 20 / 8 / 2 |

- Minimum normal force m·g/(2·mu) is 33 / 20 / 12 N at mu 0.3 / 0.5 / 0.8.
  A hold needed about 32-41 N per side, which is 1.6-2x that minimum at
  mu 0.5. The extra is needed because the pads are never exactly at the CoM
  and the box pitches.
- The arm squeeze stiffness is only about 300 N/m (PD gains). A 32 N squeeze
  therefore needs the command 12-14 cm inside the box face.
- Held boxes still creep about 2 mm/s. At mu 0.3 the grip is marginal even
  at d = 0.16, the IK limit.
- Other box heights at mu 0.5 (d 0.10 / 0.14): box bottom 0.60 and 0.90 were
  held.
- Without arm gravity feedforward: d 0.14 held, d 0.10 dropped.
- Commanded lift of +8 cm gave only 0-2 cm of real lift. The PD arm lags
  about 6-8 cm under a 1 kg-per-arm load.

Arm torque, held cases (max over the hold, percent of the ASSUMED limit):

| joint (limit) | sh_pitch (40) | sh_roll (40) | sh_yaw (20) | el_pitch (20) | el_yaw (10) | wr_pitch (6) |
|---|---|---|---|---|---|---|
| Nm | 7.6-12.4 | 3-12 | 7.5-12.8 | 5.9-7.3 | ≤1.3 | ≤1.3 |
| % | 19-31 | 8-29 | 38-64 | 29-37 | ≤13 | ≤21 |

The worst joint is shoulder yaw, at 64 %. Nothing saturates in simulation,
but these limits are guesses and the vendor payload rating is 0.5 kg per
arm.

## 3. Walking policy with the box (floating base, ONNX, full obs noise, 10 seeds)

These runs use the chest-hold arm command, the pads-only contact set and
the gripper masses. The table is removed after the lift.

| case | survived | pelvis pitch mean [min, max] (deg) | roll p2p (deg) | box |
|---|---|---|---|---|
| no box, stand 20 s | 10/10 | -0.1 [-1.2, +1.1] | 0.7 | - |
| no box, walk 0.3 m/s 10 s | 7/10 | +1.5 [-1.7, +19] | 9.0 | - |
| box, stand 20 s | 9/10 | +0.4 [-1.6, +21] | 2.5 | held 8/10, sinks 37-118 mm in 20 s |
| box, walk 0.3 m/s 10 s | 0/10 (falls at 2.2-2.8 s) | +5.0 [-0.9, +23] | 8.8 | lost |

- While walking the policy overshoots speed: 0.48 m/s with no box and
  0.63 m/s with the box, against a 0.3 m/s command. It leans forward and
  falls.
- The arm pose plus the gripper mass alone already costs 3/10 walking
  episodes.

## 4. Place (pick at box bottom 0.75 m, then lower onto a new table and release)

| table top (m) | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 | 1.05 |
|---|---|---|---|---|---|---|---|
| fixed pelvis | fail | fail | fail (4.5 deg) | ok | ok | no reach | no reach |
| policy standing (5 seeds) | 0/5 | 0/5 | 4/5 | 4/5 | 4/5 | no reach | no reach |

- Low tables fail because the reachable grip point moves 9-13 cm behind the
  CoM as the arms reach down. The box pitches forward in the grip and lands
  on its front edge (tilt 17-37 deg).
- High tables fail because the arms cannot raise the loaded box: the command
  reaches a box bottom of 1.105 m, but the box only gets to 0.88-0.90 m.
- Placement worked for 0.65-0.85 m tables only.

## 5. What the RL environment must model

- **Forearm and hand collision geometry.** The forearm, not the gripper,
  touches first unless the wrist is angled. A pads-only model hides the main
  failure.
- **Contact patch and torsional resistance at the grip.** With point
  contacts the box pivots about the squeeze axis. Pick condim/patch values
  deliberately and randomise them. Also randomise friction 0.3-0.8; 0.3 is
  marginal.
- **Force-level arm control.** Add gravity and load feedforward, or
  impedance/force control, or let the policy own the arm targets. With the
  current PD arms (~300 N/m) the squeeze force is set by penetration depth
  and a 2 kg load sags about 6-8 cm.
- **Payload in the locomotion policy.** Train with 0.86 kg of grippers plus
  a 0-2 kg box at the hands, in the chest-hold arm poses. The current policy
  cannot walk with the box (0/10) and drifts when standing.
- **Carry height.** Keep the box near a 0.65-0.85 m bottom (grip at the CoM
  x) and against the chest.
- **Real arm limits.** Measure the arm torque limits and payload rating. The
  sim uses 1 kg per arm against a 0.5 kg vendor rating.
