"""Run the trained X1 policy (policy/student.onnx) in Isaac Sim 6.0.

Everything the policy needs is read from the sidecar (policy/student.json):
joint order, default pose, action scaling, PD gains, torque limits, passive
damping, armature, rates, observation layout and normalisation, gait clock.
The control loop itself lives in x1_controller.py.

Run with Isaac Sim's Python in a clean environment (an active conda env on
LD_LIBRARY_PATH breaks Isaac's libstdc++):

  ISO="env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin DISPLAY=$DISPLAY"
  # headless evaluation, the same scenarios as the MuJoCo battery:
  $ISO ~/isaacsim/python.sh deploy/isaac/run_x1_policy.py --eval --episodes 20 --obs_noise
  # window: scripted demo (stand, walk, turn, arm poses, pushes), looped:
  $ISO ~/isaacsim/python.sh deploy/isaac/run_x1_policy.py --demo
  # window: one fixed command and arm pose:
  $ISO ~/isaacsim/python.sh deploy/isaac/run_x1_policy.py --cmd 0.5 0 0 --arms forward
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)                                     # x1_controller.py

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default=os.path.join(REPO, "policy", "student.onnx"))
ap.add_argument("--usd", default=os.path.join(HERE, "usd", "x1_isaac", "x1_isaac.usda"))
ap.add_argument("--eval", action="store_true", help="headless battery (survival per scenario)")
ap.add_argument("--scenarios", default="all", help="comma list of scenario names for --eval")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--demo", action="store_true", help="GUI demo sequence")
ap.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
ap.add_argument("--arms", default="down")
ap.add_argument("--obs_noise", action="store_true", help="full trained sensor noise")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--headless", action="store_true")
ap.add_argument("--contact_offset", type=float, default=None,
                help="explicit PhysX contactOffset (m) for the sole spheres; default = PhysX auto")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": bool(args.eval or args.headless)})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.objects import GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.experimental.utils.stage import define_prim  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from pxr import UsdPhysics, UsdShade  # noqa: E402

# ------------------------------------------------------------------ contract
from x1_controller import X1Controller, X1Policy, arms  # noqa: E402

POLICY = X1Policy(args.onnx, obs_noise=args.obs_noise, seed=args.seed)
DT = POLICY.dt
BASE_Z = POLICY.base_z

# ------------------------------------------------------------------ scenarios
def scenario(cmd=(0, 0, 0), pushes=(), cmd_sched=(), arm_sched=(), arms_pose="down", T=20.0):
    return dict(cmd=np.array(cmd, float), pushes=list(pushes), cmd_sched=list(cmd_sched),
                arm_sched=list(arm_sched) or [(0.0, arms(arms_pose))], T=T)


SCENARIOS = {
    "a_stand_still": scenario(),
    "b_stand_pushes": scenario(pushes=[(5.0, (40, 20, 0)), (10.0, (-40, -20, 0)), (15.0, (40, -20, 0))]),
    "c_walk_forward": scenario((0.8, 0, 0)),
    "d_walk_backward": scenario((-0.3, 0, 0)),
    "e_rotate_in_place": scenario((0, 0, 0.5)),
    "f_sidestep": scenario((0, 0.3, 0)),
    "g_turn_while_walking": scenario((0.5, 0, 0.4)),
    "h_walk_pushes": scenario((0.5, 0, 0), pushes=[(6.0, (0, 35, 0)), (12.0, (-35, 0, 0))]),
    "j_walk_arms_forward": scenario((0.5, 0, 0), arms_pose="forward"),
    "k_walk_arms_sideways": scenario((0.5, 0, 0), arms_pose="sideways"),
    "l_walk_arms_overhead": scenario((0.5, 0, 0), arms_pose="overhead"),
    "m_walk_arms_moving": scenario((0.5, 0, 0), arm_sched=[
        (0.0, arms("down")), (2.0, arms("down")), (3.0, arms("forward")), (6.0, arms("forward")),
        (7.0, arms("sideways")), (10.0, arms("sideways")), (11.0, arms("overhead")),
        (14.0, arms("overhead")), (15.0, arms("carry")), (18.0, arms("carry")), (19.0, arms("down"))]),
    "i_stop_and_go": scenario(cmd_sched=[(0.0, (0, 0, 0)), (3.0, (0, 0, 0)), (3.5, (0.6, 0, 0)),
                                         (11.0, (0.6, 0, 0)), (11.5, (0, 0, 0))]),
}
DEMO = scenario(T=60.0, cmd_sched=[
    (0, (0, 0, 0)), (3, (0, 0, 0)), (4, (0.5, 0, 0)), (12, (0.5, 0, 0)), (13, (0.4, 0, 0.4)),
    (20, (0.4, 0, 0.4)), (21, (0, 0.3, 0)), (26, (0, 0.3, 0)), (27, (0.5, 0, 0)),
    (50, (0.5, 0, 0)), (51, (0, 0, 0))],
    arm_sched=[(0, arms("down")), (27, arms("down")), (28, arms("forward")), (33, arms("forward")),
               (34, arms("sideways")), (39, arms("sideways")), (40, arms("overhead")),
               (45, arms("overhead")), (46, arms("down"))],
    pushes=[(54, (40, 0, 0)), (57, (0, -40, 0))])


def interp(sched, t):
    if t <= sched[0][0]:
        return np.asarray(sched[0][1], float)
    for (t0, a), (t1, b) in zip(sched, sched[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-9)
            return (1 - u) * np.asarray(a, float) + u * np.asarray(b, float)
    return np.asarray(sched[-1][1], float)


# ------------------------------------------------------------------ scene
define_prim("/World", "Xform")
GroundPlane("/World/Ground")
stage = omni.usd.get_context().get_stage()
# friction 1.0 on floor and soles, as in training (PhysX default is 0.5)
mat = UsdShade.Material.Define(stage, "/World/Physics/Friction1")
pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
pm.CreateStaticFrictionAttr(1.0)
pm.CreateDynamicFrictionAttr(1.0)
pm.CreateRestitutionAttr(0.0)
robot_prim = define_prim("/World/X1", "Xform")
robot_prim.GetReferences().AddReference(args.usd)
for prim in [stage.GetPrimAtPath("/World/Ground")] + list(stage.Traverse()):
    if prim.GetPath().pathString.startswith(("/World/Ground", "/World/X1")) and \
            prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.strongerThanDescendants, "physics")
if args.contact_offset is not None:
    from pxr import PhysxSchema
    for prim in stage.Traverse():
        if "_sole_" in prim.GetName() and prim.HasAPI(UsdPhysics.CollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(args.contact_offset)
            api.CreateRestOffsetAttr(0.0)
# Joint drives must be inert: the PD (x1_controller.py) is applied as effort. The URDF
# importer writes <dynamics damping> into the drive (USD units: N·m·s per
# DEGREE, i.e. 57x the trained damping) and ignores the zero-gain override;
# Articulation.set_dof_gains(0) reads back 0 but does not remove it.
for prim in stage.Traverse():
    if prim.GetPath().pathString.startswith("/World/X1") and prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive.GetStiffnessAttr().Set(0.0)
        drive.GetDampingAttr().Set(0.0)
torso_path = next(p.GetPath().pathString for p in stage.Traverse()
                  if p.GetPath().pathString.startswith("/World/X1") and p.GetName() == "torso_link")

define_prim("/World/PhysicsScene", "PhysicsScene")
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(DT)
RenderingManager.set_dt(0.02)

robot = Articulation("/World/X1", positions=[[0.0, 0.0, BASE_Z]], reset_xform_op_properties=True)
torso = RigidPrim(torso_path)

# ------------------------------------------------------------------ controller
class ScenarioRunner:
    """Feeds a scenario's command / arm schedule and pushes to X1Controller."""

    def __init__(self):
        self.ctl = X1Controller(robot, POLICY)
        self.ready = False
        self.alive = False

    def setup(self):
        self.ctl.setup()
        self.ready = True

    def reset(self, sc):
        self.sc = sc
        self.ctl.reset((0.0, 0.0), 0.0, interp(sc["arm_sched"], 0.0))
        self.alive, self.t, self.t_fall = True, 0.0, None

    def physics_step(self):
        if not self.alive:
            return
        t = self.ctl.t
        cmd = interp(self.sc["cmd_sched"], t) if self.sc["cmd_sched"] else self.sc["cmd"]
        self.alive = self.ctl.physics_step(cmd, interp(self.sc["arm_sched"], t))
        if not self.alive:
            self.t_fall = self.ctl.t_fall
            return
        for tp, f in self.sc["pushes"]:
            if tp <= t < tp + 0.2:
                torso.apply_forces(np.asarray(f, float)[None])
        self.t = self.ctl.t


ctl = ScenarioRunner()
plan = []
if args.eval:
    names = list(SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")
    plan = [(n, e) for n in names for e in range(args.episodes)]
results = {}


def on_physics_step(step_size, context):
    if not ctl.ready:
        return
    ctl.physics_step()


SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
omni.timeline.get_timeline_interface().play()
simulation_app.update()
ctl.setup()

if args.eval:
    for name, ep in plan:
        ctl.reset(SCENARIOS[name])
        while ctl.alive and ctl.t < ctl.sc["T"]:
            simulation_app.update()
        results.setdefault(name, []).append(ctl.alive)
        if ep == args.episodes - 1:
            s = results[name]
            print(f"[isaac] {name:22s} survival {100*np.mean(s):5.1f}% ({sum(s)}/{len(s)})", flush=True)
    out = os.path.join(REPO, "outputs", "eval", "isaac_battery.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"onnx": args.onnx, "obs_noise": args.obs_noise, "episodes": args.episodes,
               "survival": {k: float(np.mean(v)) for k, v in results.items()}},
              open(out, "w"), indent=2)
    print(f"[isaac] results -> {out}")
else:
    sc = DEMO if args.demo else scenario(tuple(args.cmd), arms_pose=args.arms, T=1e9)
    ctl.reset(sc)
    while simulation_app.is_running():
        simulation_app.update()
        if not ctl.alive:
            print(f"[isaac] fell at t={ctl.t_fall:.1f}s - resetting", flush=True)
            ctl.reset(sc)
        elif args.demo and ctl.t >= sc["T"]:
            ctl.reset(sc)
simulation_app.close()
