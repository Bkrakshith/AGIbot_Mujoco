"""Convert deploy/isaac/x1_isaac/x1_isaac.urdf into an Isaac Sim USD asset.

Run with Isaac Sim's Python in a CLEAN environment (a conda env on
LD_LIBRARY_PATH breaks Isaac's libstdc++):

  env -i HOME=$HOME PATH=/usr/bin:/bin ~/isaacsim/python.sh deploy/isaac/import_urdf.py

Floating base; joint drives set to force mode with ZERO stiffness/damping,
because the policy was trained with explicit torque PD computed every
physics step (deploy/isaac/run_x1_policy.py applies it as joint effort).
"""

import os
import shutil

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402

ext = omni.kit.app.get_app().get_extension_manager()
ext.set_extension_enabled_immediate("omni.scene.optimizer.core", True)
ext.set_extension_enabled_immediate("isaacsim.robot.schema", True)

from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = URDFImporterConfig()
cfg.urdf_path = os.path.join(HERE, "x1_isaac", "x1_isaac.urdf")
cfg.usd_path = os.path.join(HERE, "usd")
# the importer does not overwrite: it writes x1_isaac_1/, _2/, ... instead
shutil.rmtree(os.path.join(cfg.usd_path, "x1_isaac"), ignore_errors=True)
cfg.fix_base = False                 # floating base humanoid
cfg.merge_fixed_joints = False       # keep link names (pelvis, torso_link, ...)
cfg.allow_self_collision = False     # training had no self-collision except foot-foot
cfg.collision_from_visuals = False   # collision = the trained sole spheres only
cfg.joint_drive_type = "force"
cfg.joint_target_type = "none"
cfg.override_joint_stiffness = 0.0
cfg.override_joint_damping = 0.0
out = URDFImporter(cfg).import_urdf()

# The importer ignores the zero overrides and writes URDF <dynamics damping>
# into the drives (USD angular units: per DEGREE -> 57x too much damping).
# Zero every drive in the physics layer; passive damping is applied by the
# run script as effort, like in training.
from pxr import Sdf, Usd, UsdPhysics  # noqa: E402

for layer_path in [out] + [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.dirname(out))
                           for f in fs if f.endswith((".usda", ".usd"))]:
    stage = Usd.Stage.Open(layer_path)
    changed = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            for attr in (drive.GetStiffnessAttr(), drive.GetDampingAttr()):
                if attr.HasAuthoredValue() and attr.Get() != 0.0:
                    attr.Set(0.0)
                    changed += 1
    if changed:
        stage.GetRootLayer().Save()
        print(f"zeroed {changed} drive gains in {os.path.relpath(layer_path, HERE)}")
print(f"USD -> {out}")
simulation_app.close()
