"""Walk the X1 around the Isaac Sim Office with the keyboard.

Start it from a terminal (opens its own Isaac Sim window):

  ISO="env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin DISPLAY=$DISPLAY"
  $ISO ~/isaacsim/python.sh deploy/isaac/x1_office_teleop.py

or from inside an Isaac Sim window you already have open: Window > Script
Editor > File > Open this file > Run. Running it again rebuilds the scene.

Options (terminal only): --env USD (another environment), --spawn X Y YAW,
--obs_noise (full trained sensor noise), --test (headless self-test: stands,
walks forward for 6 s, exits 0 if the robot walked and never fell).

The script creates a new stage, saves it as deploy/isaac/scenes/x1_office.usda
(office environment + X1 + physics scene), presses Play and runs the policy
in policy/student.onnx with the controller in x1_controller.py.

Keys (hold to move; the command ramps smoothly and stays inside the range
the policy was trained on):
  UP / DOWN        walk forward / backward         (SHIFT + UP = fast)
  LEFT / RIGHT     turn left / right
  J / L            sidestep left / right
  SPACE            stop
  1..7             arm pose: down, forward, reach, sideways, overhead, back, carry
  P                push the torso (40 N for 0.2 s, random direction)
  R                reset the robot to the start position
  C                follow camera on / off
Click once inside the viewport first so it receives the keys.

Speed: the window advances 1/60 s of simulation per rendered frame. With the
office loaded, Isaac spends ~4 ms per 500 Hz physics step (the policy and
controller ~0.6 ms), so on an RTX 3070 the robot moves in slow motion
(~0.25x real time). The physics is unaffected.
"""

import builtins
import importlib
import math
import os
import sys
import time

IN_KIT = "omni.kit.app" in sys.modules      # True inside an open Isaac Sim window
ARGS = None
if not IN_KIT:                                # started with python.sh: open our own app
    import argparse

    ap = argparse.ArgumentParser(description="Keyboard teleop of the X1 in the Isaac Sim office")
    ap.add_argument("--env", default=None, help="environment USD (default: the Isaac Office)")
    ap.add_argument("--spawn", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"))
    ap.add_argument("--obs_noise", action="store_true", help="full trained sensor noise")
    ap.add_argument("--test", action="store_true", help="headless self-test, then exit")
    ARGS = ap.parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": ARGS.test})

import carb  # noqa: E402
import carb.input  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.rendering_manager import ViewportManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from omni.kit.async_engine import run_coroutine  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


def _isaac_dir():
    """deploy/isaac of this repo. The Script Editor does not always set
    __file__; then set X1_ISAAC_DIR or edit the fallback below."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.environ.get("X1_ISAAC_DIR", os.path.join(os.getcwd(), "deploy", "isaac"))


def _office_usd():
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if root is None:
        raise RuntimeError("Isaac Sim assets not found: set the asset root (see README) or pass --env")
    return root + "/Isaac/Environments/Office/office.usd"


ISAAC_DIR = _isaac_dir()
ROBOT_USD = os.path.join(ISAAC_DIR, "usd", "x1_isaac", "x1_isaac.usda")
SCENE = os.path.join(ISAAC_DIR, "scenes", "x1_office.usda")
LOG = os.path.join(ISAAC_DIR, "..", "..", "outputs", "logs", "isaac_teleop.log")
SPAWN_XY, SPAWN_YAW = (-17.5, 30.5), 0.0     # open floor in the Office, ~4.8 m from any object
OBS_NOISE = False                           # True = full trained sensor noise
if ARGS is not None:
    if ARGS.spawn:
        SPAWN_XY, SPAWN_YAW = tuple(ARGS.spawn[:2]), ARGS.spawn[2]
    OBS_NOISE = ARGS.obs_noise

if ISAAC_DIR not in sys.path:
    sys.path.insert(0, ISAAC_DIR)
import x1_controller  # noqa: E402

importlib.reload(x1_controller)
from x1_controller import ARM_POSES, X1Controller, X1Policy, arms  # noqa: E402

K = carb.input.KeyboardInput
MOVE_KEYS = {  # key -> (vx, vy, yaw_rate) while held
    K.UP: (0.5, 0.0, 0.0), K.NUMPAD_8: (0.5, 0.0, 0.0),
    K.DOWN: (-0.3, 0.0, 0.0), K.NUMPAD_2: (-0.3, 0.0, 0.0),
    K.LEFT: (0.0, 0.0, 0.5), K.NUMPAD_4: (0.0, 0.0, 0.5),
    K.RIGHT: (0.0, 0.0, -0.5), K.NUMPAD_6: (0.0, 0.0, -0.5),
    K.J: (0.0, 0.3, 0.0), K.L: (0.0, -0.3, 0.0),
}
FAST_VX = 1.0
ARM_KEYS = {getattr(K, f"KEY_{i + 1}"): name for i, name in enumerate(ARM_POSES)}
ACC = np.array([1.0, 1.0, 1.5])     # command ramp limits: m/s^2, m/s^2, rad/s^2
ARM_MOVE_S = 1.5                    # arm pose transition time (trained 1-2.5 s)


def log(msg):
    line = f"[x1_teleop {time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


class Teleop:
    def __init__(self):
        self.policy = X1Policy(obs_noise=OBS_NOISE)
        self.held, self.shift = set(), False
        self.cmd = np.zeros(3)
        self.arm_from = self.arm_to = self.arm_now = arms("down")
        self.arm_t = ARM_MOVE_S
        self.push_until, self.push_force = -1.0, np.zeros(3)
        self.reset_request, self.follow = False, True
        self.cam_yaw = SPAWN_YAW
        self.ctl = None
        self.cb_id = self.key_sub = self.tl_sub = None
        self.falls, self.fell_at = 0, None

    # ----------------------------------------------------------------- scene
    def build_stage(self, stage):
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        office = stage.DefinePrim("/World/Office", "Xform")
        office.GetReferences().AddReference(ARGS.env if ARGS is not None and ARGS.env else _office_usd())
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px.CreateTimeStepsPerSecondAttr(int(round(1.0 / self.policy.dt)))   # 500 Hz, as trained
        px.CreateSolverTypeAttr("TGS")
        px.CreateEnableGPUDynamicsAttr(False)
        px.CreateBroadphaseTypeAttr("MBP")
        robot = stage.DefinePrim("/World/X1", "Xform")
        robot.GetReferences().AddReference(os.path.relpath(ROBOT_USD, os.path.dirname(SCENE)))
        # friction 1.0 on floor and soles, as in training
        mat = UsdShade.Material.Define(stage, "/World/Physics/Friction1")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(1.0)
        pm.CreateDynamicFrictionAttr(1.0)
        pm.CreateRestitutionAttr(0.0)
        n_col = 0
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if path.startswith(("/World/X1", "/World/Office")) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    mat, UsdShade.Tokens.strongerThanDescendants, "physics")
                n_col += 1
            # joint drives inert: the PD is applied as effort (see run_x1_policy.py)
            if path.startswith("/World/X1") and prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(0.0)
            # trained rotor armature in the scene itself, so it survives Stop/Play
            if path.startswith("/World/X1") and prim.GetName() in self.policy.joints:
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr(
                    float(self.policy.armature[self.policy.joints.index(prim.GetName())]))
        xf = UsdGeom.Xformable(robot)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(SPAWN_XY[0], SPAWN_XY[1], self.policy.base_z))
        log(f"stage built: office + X1, friction material on {n_col} colliders")

    async def start(self):
        try:
            await self._start()
        except Exception as e:
            import traceback
            log(f"ERROR during start: {e}\n{traceback.format_exc()}")

    async def _start(self):
        ctx = omni.usd.get_context()
        omni.timeline.get_timeline_interface().stop()
        await ctx.new_stage_async()
        os.makedirs(os.path.dirname(SCENE), exist_ok=True)
        await ctx.save_as_stage_async(SCENE)            # so relative references resolve
        stage = ctx.get_stage()
        self.build_stage(stage)
        await ctx.save_stage_async()
        log(f"scene saved -> {SCENE}")
        for _ in range(30):                              # let the office payloads load
            await omni.kit.app.get_app().next_update_async()
        SimulationManager.set_physics_sim_device("cpu")
        self.robot = Articulation("/World/X1")
        torso = next(p.GetPath().pathString for p in stage.Traverse()
                     if p.GetPath().pathString.startswith("/World/X1") and p.GetName() == "torso_link")
        self.torso = RigidPrim(torso)
        self.ctl = X1Controller(self.robot, self.policy)
        self.cb_id = SimulationManager.register_callback(self.on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
        inp = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.key_sub = inp.subscribe_to_keyboard_events(self.keyboard, self.on_key)
        # Stop rebuilds the physics view while is_physics_tensor_entity_valid()
        # may still report True: force setup() + reset() on the next Play
        self.tl_sub = omni.timeline.get_timeline_interface().get_timeline_event_stream() \
            .create_subscription_to_pop_by_type(int(omni.timeline.TimelineEventType.STOP), self.on_stop)
        self.set_camera(snap=True)
        omni.timeline.get_timeline_interface().play()
        log("playing - click the viewport, then use the arrow keys (see x1_office_teleop.py)")

    def on_stop(self, event):
        if self.ctl is not None:
            self.ctl.ready = False

    def stop(self):
        self.tl_sub = None
        omni.timeline.get_timeline_interface().stop()
        if self.cb_id is not None:
            SimulationManager.deregister_callback(self.cb_id)
            self.cb_id = None
        if self.key_sub is not None:
            carb.input.acquire_input_interface().unsubscribe_to_keyboard_events(self.keyboard, self.key_sub)
            self.key_sub = None

    # ------------------------------------------------------------- keyboard
    def on_key(self, event, *args, **kwargs):
        T = carb.input.KeyboardEventType
        self.shift = bool(event.modifiers & carb.input.KEYBOARD_MODIFIER_FLAG_SHIFT)
        if event.type == T.KEY_PRESS:
            k = event.input
            if k in MOVE_KEYS:
                self.held.add(k)
            elif k == K.SPACE:
                self.held.clear()
                self.cmd[:] = 0.0
            elif k in ARM_KEYS:
                self.arm_from, self.arm_to, self.arm_t = self.arm_now.copy(), arms(ARM_KEYS[k]), 0.0
                log(f"arms -> {ARM_KEYS[k]}")
            elif k == K.P and self.ctl is not None:
                ang = np.random.uniform(0, 2 * math.pi)
                self.push_force = 40.0 * np.array([math.cos(ang), math.sin(ang), 0.0])
                self.push_until = self.ctl.t + 0.2
                log(f"push {np.round(self.push_force, 1).tolist()} N")
            elif k == K.R:
                self.reset_request = True
            elif k == K.C:
                self.follow = not self.follow
                log(f"follow camera {'on' if self.follow else 'off'}")
        elif event.type == T.KEY_RELEASE:
            self.held.discard(event.input)
        return True

    def desired_cmd(self):
        c = np.zeros(3)
        for k in self.held:
            c += MOVE_KEYS[k]
        if self.shift and c[0] > 0:
            c[0] = FAST_VX
        return self.policy.clip_cmd(c)

    # -------------------------------------------------------------- physics
    def on_physics_step(self, dt, context):
        try:
            self._physics_step(dt)
        except Exception as e:
            import traceback
            log(f"ERROR in physics step: {e}\n{traceback.format_exc()}")
            self.stop()

    def _physics_step(self, dt):
        ctl = self.ctl
        if ctl is None:
            return
        if not self.robot.is_physics_tensor_entity_valid():   # e.g. Stop/Play pressed
            ctl.ready = False
            return
        if not ctl.ready:
            ctl.setup()
            self.do_reset()
            return
        if self.reset_request or (not ctl.alive and ctl.t - self.fell_at > 1.0):
            self.do_reset()
            return
        if not ctl.alive:
            ctl.t += dt                                   # wait 1 s on the floor, then reset
            return
        pdt = self.policy.dt
        if ctl.k % self.policy.decim == 0:                # command/arm updates at policy rate
            step = ACC * pdt * self.policy.decim
            self.cmd += np.clip(self.desired_cmd() - self.cmd, -step, step)
            self.arm_t = min(self.arm_t + pdt * self.policy.decim, ARM_MOVE_S)
            u = 0.5 - 0.5 * math.cos(math.pi * self.arm_t / ARM_MOVE_S)
            self.arm_now = (1 - u) * self.arm_from + u * self.arm_to
        if not ctl.physics_step(self.cmd, self.arm_now):
            self.falls += 1
            self.fell_at = ctl.t
            log(f"fell (#{self.falls}) at t={ctl.t:.1f}s - resetting in 1 s")
            return
        if ctl.t < self.push_until:
            self.torso.apply_forces(self.push_force[None])
        if ctl.k % 10 == 0 and self.follow:
            self.set_camera()

    def do_reset(self):
        self.reset_request = False
        self.held.clear()
        self.cmd[:] = 0.0
        self.ctl.reset(SPAWN_XY, SPAWN_YAW, self.arm_now)
        self.cam_yaw = SPAWN_YAW
        self.set_camera(snap=True)
        log("robot reset at the start position")

    def set_camera(self, snap=False):
        if self.ctl is not None and self.ctl.ready and self.ctl.alive and hasattr(self.ctl, "pos"):
            pos, R = self.ctl.pos, self.ctl.R
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            pos, yaw = np.array([SPAWN_XY[0], SPAWN_XY[1], self.policy.base_z]), SPAWN_YAW
        d = (yaw - self.cam_yaw + math.pi) % (2 * math.pi) - math.pi
        self.cam_yaw = yaw if snap else self.cam_yaw + 0.05 * d      # smooth heading follow
        back = np.array([math.cos(self.cam_yaw), math.sin(self.cam_yaw), 0.0])
        eye = pos - 2.8 * back + np.array([0.0, 0.0, 0.9])
        try:
            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye.tolist(),
                                            target=(pos + np.array([0.0, 0.0, 0.1])).tolist())
        except Exception as e:  # camera is a convenience; never break control
            carb.log_warn(f"x1_teleop camera: {e}")
            self.follow = False


def self_test(t):
    """Headless check used by --test: stand 2 s, walk forward 6 s."""
    deadline = time.time() + 600
    while (t.ctl is None or not t.ctl.ready or t.ctl.t < 2.0) and time.time() < deadline:
        simulation_app.update()
    if t.ctl is None or not t.ctl.ready:
        log("self-test FAILED: the scene did not start (see the errors above)")
        return False
    start = np.array(t.ctl.pos[:2])
    t.held.add(K.UP)
    while t.ctl.t < 8.0 and t.falls == 0 and time.time() < deadline:
        simulation_app.update()
    dist = float(np.linalg.norm(np.array(t.ctl.pos[:2]) - start))
    ok = t.falls == 0 and dist > 1.5
    log(f"self-test {'PASSED' if ok else 'FAILED'}: walked {dist:.2f} m in 6 s, falls {t.falls}")
    return ok


prev = getattr(builtins, "_x1_teleop", None)
if prev is not None:
    prev.stop()
builtins._x1_teleop = Teleop()
run_coroutine(builtins._x1_teleop.start())

if not IN_KIT:
    ok = True
    if ARGS.test:
        ok = self_test(builtins._x1_teleop)
    else:
        while simulation_app.is_running():
            simulation_app.update()
    builtins._x1_teleop.stop()
    simulation_app.close()
    sys.exit(0 if ok else 1)
