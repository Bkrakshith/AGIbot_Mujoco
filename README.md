# AgiBot X1: a stand and walk policy, trained in MuJoCo and tested in Isaac Sim

> **Status: under active development, tested in simulation only.**
> The policy in this repository has been trained and evaluated in MuJoCo and
> checked in Isaac Sim. It has not been run on a physical robot. Sim-to-real
> transfer is pending, and I will update this README with results once
> hardware testing has been done.
>
> **Disclaimer.** This software is provided "as is", without warranty of any
> kind. The author is not responsible for any sim-to-real deployment of these
> policies or of any part of this repository. Running them on real hardware
> is done entirely at the user's own risk, including any damage to the robot,
> its surroundings or people. Always follow the robot manufacturer's safety
> procedures (supervision, emergency stop, safety tether) when testing
> learned controllers.

This repository trains the AgiBot X1 humanoid to stand and walk, and to keep
walking while its arms are held in task poses (forward, sideways, overhead,
back, carrying). Training runs in MuJoCo MJX with JAX and brax PPO. The
trained policy is a small ONNX network, so you can run it without any of the
training stack. It ships in `policy/`, and you can drive the robot around
with your keyboard in either simulator:

- MuJoCo is the quickest way to see it walk. You need Python and about five
  minutes.
- Isaac Sim 6.0 is a second physics engine. I use it as a sim-to-sim check,
  and it has an office you can walk the robot around in.

The robot model and most training parameters come from AgiBot's own
[agibot_x1_train](https://github.com/AgibotTech/agibot_x1_train). Only the files
this project uses were copied in (see `assets/x1_vendor/SOURCE.md`).

## Contents

1. [Results so far](#results-so-far)
2. [What is in the repository](#what-is-in-the-repository)
3. [Quick start: run the policy in MuJoCo](#quick-start-run-the-policy-in-mujoco)
4. [Setting up Isaac Sim](#setting-up-isaac-sim)
5. [Running the robot in Isaac Sim](#running-the-robot-in-isaac-sim)
6. [How the controller works](#how-the-controller-works)
7. [Training your own policy](#training-your-own-policy)
8. [Troubleshooting](#troubleshooting)
9. [Sim-to-real status](#sim-to-real-status)

## Results so far

All numbers below are for the released policy, `policy/student.onnx`, with
the full X1 sensor noise switched on. A scenario "survives" if the robot is
still upright after 20 seconds.

| scenario | MuJoCo (50 runs each) | Isaac Sim (4 runs each) |
|---|---|---|
| stand still | 100 % | 4/4 |
| stand, pushed with 40 N three times | 96 % | 3/4 |
| walk forward at 0.8 m/s | 92 % | 4/4 |
| walk backward, turn on the spot, sidestep, turn while walking | 100 % | 4/4 each |
| walk while pushed with 35 N | 96 % | 4/4 |
| walk with arms forward, sideways or overhead | 100 % | 4/4 each |
| walk while the arms move through poses | 98 % | 4/4 |
| stop and go | 100 % | 4/4 |

While walking in MuJoCo the torso pitch stays within 2 degrees on average and
the commanded speed is tracked closely (0.6 m/s commanded, 0.56 m/s measured).
In Isaac Sim at 0.5 m/s it walks at 0.51 m/s, with the same joint tracking
error and torque use as in MuJoCo.

Four runs per scenario in Isaac Sim is a sanity check, not a measurement.
Treat those numbers as "it works in a second physics engine", nothing more.

## What is in the repository

| path | what it is |
|---|---|
| `policy/` | the trained policy: `student.onnx` and its sidecar `student.json` (everything a controller needs to run it) |
| `assets/` | the X1 model. `x1_vendor/` is the imported vendor subset; `x1_mjx.xml` (training) and `x1_full.xml` (evaluation) are generated from it by `scripts/build_assets.py` |
| `configs/` | every gain, reward weight, randomisation range and curriculum stage, in YAML |
| `src/x1_locomotion/` | the environment, gait generator (CPG), rewards, PPO teacher and student distillation |
| `train/` | training scripts, plus `view.py` (interactive MuJoCo viewer) and `play.py` (renders videos) |
| `eval/` | `run_battery.py`, the 13-scenario test used for the table above |
| `export/` | converts a trained checkpoint to ONNX and writes the sidecar |
| `deploy/isaac/` | everything for Isaac Sim (see below) |
| `tests/` | unit tests (`python -m pytest tests/ -q`) |

## Quick start: run the policy in MuJoCo

You need Linux (tested on Ubuntu 22.04) and Python 3.11. A GPU is not needed
to run the policy; it is only needed for training.

```bash
git clone https://github.com/Bkrakshith/AGIbot_Mujoco.git
cd AGIbot_Mujoco
bash scripts/setup_env.sh          # creates .venv with the pinned packages (CPU only)
source .venv/bin/activate
```

If `python3.11` is not on your PATH, point the script at it:
`PY=/path/to/python3.11 bash scripts/setup_env.sh`.

### Drive the robot with the keyboard

```bash
python train/view.py --onnx policy/student.onnx --obs_noise
```

A MuJoCo window opens with the robot standing. Click inside it, then:

| key | what it does |
|---|---|
| Up / Down | forward speed +/- 0.1 m/s |
| Page Up / Page Down | sideways speed +/- 0.1 m/s |
| Left / Right | turning speed +/- 0.1 rad/s |
| Home | stop |
| End | push the torso (40 N for 0.2 s, random direction) |
| Insert | reset the robot |
| 1 to 7 | arm pose: down, forward, reach, sideways, overhead, back, carry |
| 8 | random arm poses, like in training |

`--obs_noise` adds the same sensor noise the policy was trained with. Leave it
out if you want to see the policy on perfect sensors.

### Run the test battery or record videos

```bash
python eval/run_battery.py --onnx policy/student.onnx --obs_noise --episodes 50
python train/play.py --onnx policy/student.onnx --scenario c_walk_forward --obs_noise
```

The battery takes a while on CPU (13 scenarios x 50 runs x 20 s). Use
`--episodes 4` for a quick check. Results are written to `outputs/eval/`,
videos to `outputs/videos/`.

## Setting up Isaac Sim

Everything Isaac-specific lives in `deploy/isaac/`:

| file | what it does |
|---|---|
| `x1_isaac/x1_isaac.urdf` + `meshes/` | the exact robot the policy was trained on, as a URDF (already generated) |
| `make_isaac_urdf.py` | regenerates that URDF from `assets/x1_full.xml`; only needed if you change the robot model |
| `import_urdf.py` | converts the URDF into a USD file Isaac Sim can load |
| `x1_controller.py` | the policy and joint controller, driven entirely by `policy/student.json` |
| `x1_office_teleop.py` | keyboard control in the Isaac office environment |
| `run_x1_policy.py` | the test battery, a scripted demo, or one fixed command on an empty floor |

These steps were tested with Isaac Sim 6.0.0 (release candidate 59) on
Ubuntu 22.04 with an RTX 3070 (8 GB). Isaac Sim needs an NVIDIA RTX GPU.

### Step 1: install Isaac Sim 6.0

Download the Isaac Sim 6.0 standalone package for Linux from NVIDIA's Isaac
Sim documentation site (https://docs.isaacsim.omniverse.nvidia.com, under
Installation > Download). Unzip it to `~/isaacsim`:

```bash
mkdir -p ~/isaacsim
unzip ~/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64.zip -d ~/isaacsim   # your file name may differ
cd ~/isaacsim
./post_install.sh
./isaac-sim.compatibility_check.sh     # checks your GPU and driver
```

The commands in this README assume `~/isaacsim`. If you put it somewhere
else, change that path wherever it appears.

### Step 2: get the Isaac Sim assets

The office environment is part of NVIDIA's asset pack. You have two options:

- **Local copy (recommended).** Download the asset pack for 6.0 from the same
  download page and unzip it to `~/isaacsim_assets`, so that the folder
  `~/isaacsim_assets/Assets/Isaac/6.0/Isaac/Environments/Office` exists. Then
  start Isaac Sim once with the asset location on the command line. Isaac
  remembers it after that:

  ```bash
  cd ~/isaacsim
  ./isaac-sim.sh --/persistent/isaac/asset_root/default="$HOME/isaacsim_assets/Assets/Isaac/6.0"
  ```

  Close the window once it has opened.

- **NVIDIA's cloud assets.** Skip the download and Isaac Sim fetches the
  assets on demand. This works, but the first start of the office takes much
  longer.

### Step 3: use a clean shell for Isaac Sim

If you use conda, its `libstdc++` can shadow the one Isaac Sim ships with and
Isaac fails to start (`GLIBCXX_... not found` or similar). The simplest fix is
to start every Isaac command from an empty environment. Define this once per
terminal:

```bash
ISO="env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin DISPLAY=$DISPLAY"
```

Every Isaac command below starts with `$ISO`. If you do not use conda you can
leave it out, but it does no harm.

### Step 4: install onnxruntime for Isaac Sim's Python

Isaac Sim has its own Python (3.12) without onnxruntime. Install it into a
folder inside this repository so it does not touch your Isaac Sim install:

```bash
cd /path/to/AGIbot_Mujoco
$ISO ~/isaacsim/python.sh -m pip install --target deploy/isaac/_pydeps onnxruntime
```

### Step 5: convert the robot to USD

```bash
$ISO ~/isaacsim/python.sh deploy/isaac/import_urdf.py
```

This writes `deploy/isaac/usd/x1_isaac/x1_isaac.usda` and should end with
`zeroed 24 drive gains` and `USD -> .../x1_isaac.usda`. Run it again whenever
the URDF changes; it replaces the old USD.

### Step 6: check that everything works

```bash
$ISO ~/isaacsim/python.sh deploy/isaac/x1_office_teleop.py --test
```

This loads the office without opening a window, lets the robot stand for two
seconds, then walks it forward for six. It should finish with a line like
`self-test PASSED: walked 2.68 m in 6 s, falls 0`. The first run takes a few
minutes because Isaac Sim compiles its shaders and loads the office.

## Running the robot in Isaac Sim

All commands are run from the repository folder, with `ISO` defined as in
step 3.

### Walk around the office with the keyboard

```bash
$ISO ~/isaacsim/python.sh deploy/isaac/x1_office_teleop.py
```

Isaac Sim opens, builds a new scene (the office, the X1 and a physics scene),
saves it as `deploy/isaac/scenes/x1_office.usda` and presses Play. The robot
starts standing in an open part of the office. Click once inside the viewport
so it gets your key presses, then:

| key | what it does |
|---|---|
| Up / Down | walk forward (0.5 m/s) / backward (0.3 m/s), while held |
| Shift + Up | walk forward fast (1.0 m/s) |
| Left / Right | turn left / right |
| J / L | sidestep left / right |
| Space | stop |
| 1 to 7 | arm pose: down, forward, reach, sideways, overhead, back, carry |
| P | push the torso (40 N for 0.2 s, random direction) |
| R | put the robot back at the start |
| C | follow camera on / off (switch it off to move the camera yourself) |

Speed changes are ramped, so tapping a key gives a gentle start rather than a
jolt, and speeds never leave the range the policy was trained on. If the
robot falls it is reset automatically after a second.

Useful options:

- `--obs_noise` runs with the trained sensor noise.
- `--spawn X Y YAW` starts the robot somewhere else. Units are metres and
  radians.
- `--env some_scene.usd` swaps the office for another environment. It needs
  a floor with collision at height 0.

Two things to know about the office scene:

- It runs in slow motion, at roughly a quarter of real time on an RTX 3070.
  Isaac advances 1/60 s of simulation per rendered frame, and with the whole
  office loaded each physics step costs about 4 ms. The policy runs at 500
  physics steps per simulated second, so the window cannot keep up. The
  physics itself is unaffected; it is only slower to watch.
- Only the floor has collision. The office furniture and walls are visual
  only, so the robot walks straight through desks.

If you already have Isaac Sim open, you can also run the script from inside
it: Window > Script Editor, File > Open `deploy/isaac/x1_office_teleop.py`,
then Run. If Isaac cannot find the repository that way, set the environment
variable `X1_ISAAC_DIR` to the full path of `deploy/isaac` before starting
Isaac Sim.

### Scripted demo, fixed commands and the test battery

`run_x1_policy.py` uses a plain floor instead of the office and runs in real
time:

```bash
# a 60 s demo on loop: stand, walk, turn, sidestep, arm poses, pushes
$ISO ~/isaacsim/python.sh deploy/isaac/run_x1_policy.py --demo

# one fixed command (forward, sideways, turn) and an arm pose
$ISO ~/isaacsim/python.sh deploy/isaac/run_x1_policy.py --cmd 0.5 0 0 --arms forward

# the same 13 scenarios as the MuJoCo battery, without a window
$ISO ~/isaacsim/python.sh deploy/isaac/run_x1_policy.py --eval --episodes 4 --obs_noise
```

The battery prints a survival line per scenario and saves the results to
`outputs/eval/isaac_battery.json`. `--scenarios a_stand_still,c_walk_forward`
runs only the ones you list.

## How the controller works

If you want to run the policy somewhere else (another simulator, or
eventually the robot), everything you need is in `policy/student.json`.
`deploy/isaac/x1_controller.py` is a complete, short implementation of it,
and a good place to start.

The policy runs at 100 Hz and the joint PD loop at 500 Hz. Each policy step
outputs 12 numbers, one per leg joint, and the target angle for each joint is
`default_angle + 0.5 * action`. The 12 arm joints are not driven by the
policy. They follow an arm-pose command from the task, and the policy is told
what that command is so it can balance for it.

Every physics step, each joint gets the torque
`clip(kp * (target - q) - kd * qd, +/- torque_limit) - damping * qd`, with the
gains and limits from the sidecar. Switch the simulator's own joint drives
off (zero stiffness and damping). If you leave them on, the robot is damped
twice.

The input is the last 50 observations, each 84 numbers:
  - gravity direction in the pelvis frame (3);
  - pelvis angular velocity (3);
  - joint angles relative to the default pose (24);
  - joint velocities (24);
  - the previous action (12);
  - the walking command: forward speed, sideways speed, turning speed, pelvis height (4);
  - the arm-pose command (12);
  - the sine and cosine of a gait clock with a 0.7 s cycle (2).
Feed the observations in raw; the network does its own normalisation. After
a reset, fill the history with the published mean. Keep commands inside
-0.4 to 1.2 m/s forward, +/- 0.4 m/s sideways and +/- 0.6 rad/s turning. The
policy never saw anything outside those ranges.

## Training your own policy

Training needs an NVIDIA GPU with CUDA. I trained on an RTX 3070 with 8 GB.

```bash
CUDA=1 EXPORT=1 bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/build_assets.py
python -m pytest tests/ -q

# teacher: PPO with privileged observations, through a 7-stage curriculum
python train/train_teacher.py --run_name teacher_v1 --backend gpu

# student: learns to copy the teacher from 50 steps of sensor history only
python train/train_student.py --teacher_ckpt outputs/checkpoints/teacher_v1/ckpt_N \
       --run_name student_v1 --backend gpu

# test it, export it, test the export
python eval/run_battery.py --checkpoint outputs/checkpoints/student_v1/ckpt_best --obs_noise
python export/params_to_torch.py --checkpoint outputs/checkpoints/student_v1/ckpt_best --out outputs/onnx/student.pt
python export/export_onnx.py --torch outputs/onnx/student.pt
python export/verify_onnx.py --onnx outputs/onnx/student.onnx --torch outputs/onnx/student.pt
python eval/run_battery.py --onnx outputs/onnx/student.onnx --obs_noise --gate 0.9
```

Training the teacher takes many hours on an RTX 3070. A few choices made the
difference between a policy that works and one that doesn't.

The curriculum adds one difficulty per stage. Walking commands grow from 0.4
to 1.0 of the full range, arm poses from none to half to full, physics
randomisation from 0 to 50, 75 and then 100 %, and finally pushes of 30 to
60 N. A stage passes on its best checkpoint, not its last one.

The teacher sees the randomised physics parameters (friction, motor
strength, latency, mass, centre of mass), because the student can work those
out from its history. It does not see the push force. I tried that, and the
student learned to wait for information it never gets, then fell over when
pushed.

Only the student gets sensor noise. The teacher trains on clean sensors, and
the student gets the full X1 noise and learns its own observation encoder.

Training and evaluation use the same foot: four contact points under each
sole, as in the vendor model. When training used a box-shaped foot, the
policy learned a shuffle that fell apart on the real foot shape.

## Troubleshooting

**Isaac Sim will not start, or complains about `GLIBCXX` / `libstdc++`.**
You are probably in a conda environment. Use the clean `$ISO` prefix from
step 3.

**`ModuleNotFoundError: No module named 'onnxruntime'` in Isaac Sim.** Step
4 was skipped, or onnxruntime was installed into a different Python. It has
to be installed with `~/isaacsim/python.sh`, into `deploy/isaac/_pydeps`.

**`Isaac Sim assets not found`.** Isaac does not know where the asset pack
is. Do the one-time start from step 2, or pass `--env` with the full path to
`office.usd`.

**The robot stands but falls within a few seconds of walking.** The joint
drives in the USD are not switched off. Isaac's URDF importer copies the URDF
joint damping into the joint drives, and it counts that damping per degree,
so each joint ends up with about 57 times too much. Both Isaac scripts zero
the drives when they load the robot, and `import_urdf.py` zeroes them in the
USD. If you load the robot some other way, set every joint drive's
stiffness and damping to 0 yourself.

**Random crashes (segmentation faults) during long runs.** Some machines have
CPU cores that fail under sustained load. Mine has two (logical CPUs 8 and
9). If you see the same, keep processes off the bad cores with `taskset`,
for example `taskset -c 0-7,10-31 python train/train_teacher.py ...`.

**The office scene is in slow motion.** That is expected; see the notes in
"Walk around the office with the keyboard". `run_x1_policy.py` on a plain
floor runs in real time.

## Sim-to-real status

The robot has not been tested on hardware. Before any hardware test, these
effects still need to be added to training (the vendor's X1 training setup
includes all of them):

- differences in gain and strength from joint to joint;
- encoder zero offsets;
- joint friction;
- sensor latency.

The arm torque limits and gains are my own estimates, not vendor values.
Hardware results will be added here once testing has been done.
