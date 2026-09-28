"""Headless check of the office colliders (office_collision.py).

  ISO="env -i HOME=$HOME USER=$USER PATH=/usr/local/bin:/usr/bin:/bin TERM=xterm"
  $ISO taskset -c 0-7,10-31 ~/isaacsim/python.sh deploy/isaac/check_office_collision.py
  # the office colliders as shipped, for comparison:
  $ISO taskset -c 0-7,10-31 ~/isaacsim/python.sh deploy/isaac/check_office_collision.py --no_office_collision
  # timing reference without the office (ground plane only, no object checks):
  $ISO taskset -c 0-7,10-31 ~/isaacsim/python.sh deploy/isaac/check_office_collision.py --empty

1. drops the 2 kg corrugated box onto the reception desks and meeting tables
   of office_map.yaml, and a 0.2 m cube onto a chair; reports where they come
   to rest against the surface found by a downward ray,
   A cube pushed along the conference table at 1 m/s checks the friction
   (default material of the scene, mu = 1 -> slides ~5 cm; PhysX default 0.5
   -> ~10 cm),
2. checks that the robot standing spots of office_map.yaml are free,
3. walks the X1 (policy, 0.5 m/s) into a wall and into the reception counter
   and reports how close its base got. The X1 collides only through its
   trained sole spheres, so its body walks through furniture; --bumper adds
   a massless box collider to the torso (in this check only) to show that
   the office itself blocks,
4. times the physics: PhysX time per 500 Hz step and the whole loop per step
   (policy and controller included) with the robot standing at the spawn.
Prints "check PASSED" if every object rests on its surface, every standing
spot is free and (with --bumper) the robot never passed an obstacle.
"""

import argparse
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--no_office_collision", action="store_true", help="office colliders as shipped")
ap.add_argument("--empty", action="store_true", help="no office, ground plane only (timing reference)")
ap.add_argument("--bumper", action="store_true", help="box collider on the torso for the walk tests")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402
from omni.physx.scripts.ifaces import get_physxunittests_interface  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from office_collision import add_office_collision  # noqa: E402
from x1_controller import X1Controller, X1Policy, arms  # noqa: E402

ASSETS = get_assets_root_path()
OFFICE_USD = ASSETS + "/Isaac/Environments/Office/office.usd"
BOX_USD = ASSETS + ("/Isaac/Environments/warehouse_20x20_envi/props/sm_box_corrugated_brown_b12_01/"
                    "sm_box_corrugated_brown_b12_01.usd")
ROBOT_USD = os.path.join(HERE, "usd", "x1_isaac", "x1_isaac.usda")
MAP = yaml.safe_load(open(os.path.join(HERE, "office_map.yaml")))
SPAWN = MAP["robot_spawn"]
# robot walks: name -> (start x, y, yaw, obstacle description, axis, obstacle face coordinate)
WALKS = {
    "wall": (-21.2, 30.5, math.pi, "hall west window blinds x=-23.2", 0, -23.2),
    "counter": (0.15, -1.0, math.pi, "reception_main counter x=-1.87", 0, -1.873),
}
DROP = 0.05                     # m above the surface


def log(msg):
    print(f"[check] {msg}", flush=True)


# ------------------------------------------------------------------ scene
policy = X1Policy()
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
if args.empty:
    PhysicsSchemaTools.addGroundPlane(stage, "/World/Office/GroundPlane", "Z", 50.0, Gf.Vec3f(0.0), Gf.Vec3f(0.5))
else:
    stage.DefinePrim("/World/Office", "Xform").GetReferences().AddReference(OFFICE_USD)
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px.CreateTimeStepsPerSecondAttr(int(round(1.0 / policy.dt)))
px.CreateSolverTypeAttr("TGS")
px.CreateEnableGPUDynamicsAttr(False)
px.CreateBroadphaseTypeAttr("MBP")
mat = UsdShade.Material.Define(stage, "/World/Physics/Friction1")
pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
pm.CreateStaticFrictionAttr(1.0)
pm.CreateDynamicFrictionAttr(1.0)
pm.CreateRestitutionAttr(0.0)
if not (args.no_office_collision or args.empty):
    log(f"office collision: {dict(add_office_collision(stage, '/World/Office', mat))}")
robot = stage.DefinePrim("/World/X1", "Xform")
robot.GetReferences().AddReference(ROBOT_USD)
for prim in stage.Traverse():
    path = prim.GetPath().pathString
    if path.startswith(("/World/X1", "/World/Office")) and prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    if path.startswith("/World/X1") and prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive.GetStiffnessAttr().Set(0.0)
        drive.GetDampingAttr().Set(0.0)
    if path.startswith("/World/X1") and prim.GetName() in policy.joints:
        PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr(
            float(policy.armature[policy.joints.index(prim.GetName())]))
xf = UsdGeom.Xformable(robot)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(SPAWN[0], SPAWN[1], policy.base_z))
if args.bumper:
    torso = next(p for p in stage.Traverse()
                 if p.GetPath().pathString.startswith("/World/X1") and p.GetName() == "torso_link")
    bumper = UsdGeom.Cube.Define(stage, torso.GetPath().AppendChild("check_bumper"))
    bumper.CreateSizeAttr(1.0)
    bumper.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.1))
    bumper.AddScaleOp().Set(Gf.Vec3f(0.25, 0.4, 0.5))
    bumper.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(bumper.GetPrim())
    UsdPhysics.MassAPI.Apply(bumper.GetPrim()).CreateMassAttr(1e-4)

# test objects: (name, prim path, x, y, expected surface z, offset of the prim origin above its bottom)
tests = []
places = {} if args.empty else dict(MAP["reception_desks"])
if not args.empty:
    places.update({k: v for k, v in MAP["meeting_tables"].items() if k in ("meeting_east_table", "conference_table")})
for name, d in places.items():
    x, y, z = d["place"]
    p = stage.DefinePrim(f"/World/Test/box_{name}", "Xform")
    p.GetReferences().AddReference(BOX_USD)              # origin at the bottom face
    p.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z + DROP))
    tests.append((name, p.GetPath().pathString, x, y, z, 0.0))
# a cube onto the chair nearest to the east meeting table
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
chairs = [p for p in stage.GetPrimAtPath("/World/Office").GetChildren() if p.GetName().startswith("SM_Chair")]
chair = None if args.empty else min(chairs, key=lambda p: np.linalg.norm(
    np.array(cache.ComputeWorldBound(p).ComputeAlignedRange().GetMidpoint())[:2] - [2.1, 38.7]))
if chair is not None:
    cr = cache.ComputeWorldBound(chair).ComputeAlignedRange()
    cx, cy = cr.GetMidpoint()[0], cr.GetMidpoint()[1]
    cube = UsdGeom.Cube.Define(stage, "/World/Test/cube_chair")
    cube.CreateSizeAttr(0.2)
    cube.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cr.GetMax()[2] + DROP))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)
    tests.append((f"chair {chair.GetName()}", "/World/Test/cube_chair", cx, cy, None, 0.1))
    slide = UsdGeom.Cube.Define(stage, "/World/Test/cube_slide")
    slide.CreateSizeAttr(0.2)
    slide.AddTranslateOp().Set(Gf.Vec3d(2.3, 15.5, 0.85))
    UsdPhysics.CollisionAPI.Apply(slide.GetPrim())
    UsdPhysics.MassAPI.Apply(slide.GetPrim()).CreateMassAttr(1.0)
    UsdPhysics.RigidBodyAPI.Apply(slide.GetPrim()).CreateVelocityAttr(Gf.Vec3f(0.0, 1.0, 0.0))

# ------------------------------------------------------------------ run
SimulationManager.set_physics_sim_device("cpu")
art = Articulation("/World/X1")
ctl = X1Controller(art, policy)
state = {"cmd": np.zeros(3), "t_pre": 0.0, "physx": [], "n": 0}


def on_pre(dt, context):
    state["t_pre"] = time.perf_counter()


def on_post(dt, context):
    state["physx"].append(time.perf_counter() - state["t_pre"])
    state["n"] += 1
    if ctl.ready and ctl.alive:
        ctl.physics_step(state["cmd"], arms("down"))


SimulationManager.register_callback(on_pre, IsaacEvents.PRE_PHYSICS_STEP)
SimulationManager.register_callback(on_post, IsaacEvents.POST_PHYSICS_STEP)
omni.timeline.get_timeline_interface().play()
simulation_app.update()
ctl.setup()
ctl.reset(SPAWN[:2], SPAWN[2])
stats = get_physxunittests_interface().get_physics_stats()
log(f"PhysX: {stats['numStaticRigids']} static actors, {stats['numTriMeshShapes']} triangle meshes, "
    f"{stats['numConvexShapes']} convex hulls")
sq = get_physx_scene_query_interface()


def surface(x, y):
    """Highest office collider under (x, y), ignoring the test objects."""
    hits = []
    sq.raycast_all(Gf.Vec3d(x, y, 2.5), Gf.Vec3d(0, 0, -1), 5.0,
                   lambda h: hits.append((h.position[2], h.collision)) or True)
    hits = [h for h in hits if h[1].startswith("/World/Office")]
    return max(hits) if hits else (None, None)


def run(seconds):
    t_end = ctl.t + seconds
    while ctl.t < t_end and ctl.alive:
        simulation_app.update()


ok = True
# 1 + 4: settle the objects with the robot standing, then time 4 s of standing
run(3.0)
mode = "empty" if args.empty else "shipped" if args.no_office_collision else "office_collision"
for _ in range(3):
    state["physx"], n0, w0 = [], state["n"], time.perf_counter()
    run(4.0)
    wall = (time.perf_counter() - w0) / max(state["n"] - n0, 1)
    log(f"timing ({mode}): PhysX {1e3 * np.median(state['physx']):.2f} ms/step (median), "
        f"whole loop {1e3 * wall:.2f} ms/step, {1.0 / wall:.0f} steps/s ({policy.dt / wall:.2f}x real time)")
for name, path, x, y, z_exp, off in tests:
    pos = RigidPrim(path).get_world_poses()[0].numpy()[0]
    z_ray, hit = surface(x, y)
    bottom = pos[2] - off
    drift = float(np.hypot(pos[0] - x, pos[1] - y))
    good = z_ray is not None and abs(bottom - z_ray) < 0.03 and bottom > 0.3 and drift < 0.1
    ok &= good
    log(f"{name:24s} rests at z={bottom:.3f} (surface {z_ray:.3f} {hit.split('/')[3] if hit else ''}"
        f"{', map ' + format(z_exp, '.3f') if z_exp is not None else ''}), drift {drift:.3f} m "
        f"{'OK' if good else 'BAD'}")

if not args.empty:
    sx, sy, _ = RigidPrim("/World/Test/cube_slide").get_world_poses()[0].numpy()[0]
    d = float(np.hypot(sx - 2.3, sy - 15.5))
    log(f"cube pushed at 1 m/s on the conference table slid {d:.3f} m (mu ~ {1.0 / (2 * 9.81 * max(d, 1e-3)):.2f})")

# 2: standing spots
for group in () if args.empty else ("reception_desks", "meeting_tables"):
    for name, d in MAP[group].items():
        for key in ("stand", "stand_alt"):
            if key in d:
                x, y, _ = d[key]
                hits = []
                sq.overlap_box(Gf.Vec3f(0.28, 0.28, 0.55), Gf.Vec3f(x, y, 0.63), Gf.Vec4f(0, 0, 0, 1),
                               lambda h: hits.append(h.collision) or True, False)
                hits = [h for h in hits if not h.startswith(("/World/X1", "/World/Test"))]
                ok &= not hits
                log(f"{name}.{key} ({x}, {y}) {'free' if not hits else 'BLOCKED by ' + str(hits[:3])}")

# 3: walk into a wall and into the counter
for name, (x, y, yaw, what, axis, face) in ({} if args.empty else WALKS).items():
    ctl.reset((x, y), yaw)
    state["cmd"] = np.zeros(3)
    run(1.5)
    state["cmd"] = np.array([0.5, 0.0, 0.0])
    start = ctl.pos.copy()
    sgn = math.copysign(1.0, face - start[axis])
    closest, t_end = abs(face - start[axis]), ctl.t + 8.0
    while ctl.t < t_end and ctl.alive:
        simulation_app.update()
        closest = min(closest, sgn * (face - ctl.pos[axis]))
    state["cmd"] = np.zeros(3)
    passed = closest < 0.0
    ok &= not (passed and args.bumper)
    log(f"walk into {what}: start {abs(face - start[axis]):.2f} m away, base got to {closest:.2f} m from it, "
        f"{'PASSED THROUGH' if passed else 'blocked'}, "
        f"{'fell at t=%.1f s' % (ctl.t_fall - 1.5) if not ctl.alive else 'still standing'}")

log(f"check {'PASSED' if ok else 'FAILED'}")
omni.timeline.get_timeline_interface().stop()
simulation_app.close()
sys.exit(0 if ok else 1)
