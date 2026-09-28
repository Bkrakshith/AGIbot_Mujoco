"""Box pick-and-place props and grasping for the X1 in the Isaac Sim office.

Used by x1_office_teleop.py with --task:
  - a 0.75 m pickup table beside the north reception (the reception counter's
    visitor ledge is 1.12 m, above the arms' reach, and the desk behind it is
    enclosed by the counter), with the handle box on it;
  - a simple gripper block on each wrist (the OmniPicker has no public model);
  - grasping: when both grippers close within GRASP_TOL of the handles, the
    box glides over GRASP_BLEND_S into the held pose (handles centred in the
    grippers) and is then fixed to the hands (kinematic) until they open.
    This is a visual stand-in for the 30 N pinch; the gripper fingers are not
    simulated. The walking policy was trained with 0-0.3 kg at the hands.

The box design (size, handles, hold pose) is manip/handle_box.yaml.
"""

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOX_USD = os.path.join(HERE, "props", "handle_box.usda")
BOX_VISUAL = "/Isaac/Environments/warehouse_20x20_envi/props/sm_box_corrugated_brown_b12_01/sm_box_corrugated_brown_b12_01.usd"
BOX_HALF_H = 0.07

# world frame (m, rad); the robot faces -y at the pickup table
# the box sits 0.12 m from the table edge nearest the robot, so the pelvis
# stands 0.37 m from the table (the gait drifts a few cm when it stops)
PICK_TABLE = dict(centre=(-15.0, 26.87), size=(0.9, 0.5), top=0.75)     # size = (x, y)
BOX_PICK = (-15.0, 27.00, -math.pi / 2)                                 # box centre xy + yaw
# gripper centre point in the wrist_roll frame (pad from 0.03 to 0.15 m along -y)
TCP_IN_WRIST = np.array([0.0, -0.10, 0.0])
HANDLE_Y = 0.2218            # handle centres at +-y in the box frame
HANDLE_Z = 0.045
GRASP_TOL = 0.20             # m, each gripper to its handle when the grippers close
GRASP_BLEND_S = 0.6          # s, box glides into the grippers


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _R_to_quat(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


def add_task_props(stage, assets_root, material=None):
    """Pickup table + handle box, authored before Play."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cx, cy = PICK_TABLE["centre"]
    sx, sy = PICK_TABLE["size"]
    top = PICK_TABLE["top"]
    table = UsdGeom.Cube.Define(stage, "/World/Task/PickupTable")
    table.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(table)
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, top / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, top))
    table.GetDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.40, 0.25)])
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    box = stage.DefinePrim("/World/Task/Box", "Xform")
    box.GetReferences().AddReference(BOX_USD)
    stage.GetPrimAtPath("/World/Task/Box/Visual").GetReferences().AddReference(assets_root + BOX_VISUAL)
    bx, by, byaw = BOX_PICK
    bxf = UsdGeom.Xformable(box)
    bxf.ClearXformOpOrder()
    bxf.AddTranslateOp().Set(Gf.Vec3d(bx, by, top + BOX_HALF_H + 0.002))
    bxf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, math.degrees(byaw)))
    if material is not None:
        for p in (table.GetPrim(), stage.GetPrimAtPath("/World/Task/Box")):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")


def add_gripper_visuals(stage):
    """A dark block per wrist where the OmniPicker sits (visual only)."""
    from pxr import Gf, UsdGeom

    for side in ("left", "right"):
        wrist = next((p for p in stage.Traverse() if p.GetPath().pathString.startswith("/World/X1")
                      and p.GetName() == f"{side}_wrist_roll"), None)
        if wrist is None:
            continue
        g = UsdGeom.Cube.Define(stage, wrist.GetPath().AppendChild("gripper_visual"))
        g.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.09, 0.0))
        xf.AddScaleOp().Set(Gf.Vec3f(0.08, 0.12, 0.06))
        g.GetDisplayColorAttr().Set([Gf.Vec3f(0.12, 0.12, 0.12)])


class Grasp:
    """Close/open the grippers on the box handles."""

    def __init__(self, stage):
        from isaacsim.core.experimental.prims import RigidPrim

        wrists = [next(p.GetPath().pathString for p in stage.Traverse()
                       if p.GetPath().pathString.startswith("/World/X1") and p.GetName() == f"{s}_wrist_roll")
                  for s in ("left", "right")]
        self.wrists = RigidPrim(wrists)
        self.box = RigidPrim("/World/Task/Box")
        self.box_prim = stage.GetPrimAtPath("/World/Task/Box")
        self.held = False
        self.offset = None          # box pose in the left wrist frame while held
        self.closed = False
        self.blend = None           # (start pose, steps left) while gliding into the grip

    def box_pose(self):
        p, q = self.box.get_world_poses()
        return p.numpy()[0], _quat_to_R(q.numpy()[0])

    def tcps(self):
        p, q = self.wrists.get_world_poses()
        p, q = p.numpy(), q.numpy()
        return [p[i] + _quat_to_R(q[i]) @ TCP_IN_WRIST for i in range(2)], p, q

    def handle_errors(self):
        bp, bR = self.box_pose()
        (tl, tr), _, _ = self.tcps()
        hl = bp + bR @ np.array([0.0, HANDLE_Y, HANDLE_Z])
        hr = bp + bR @ np.array([0.0, -HANDLE_Y, HANDLE_Z])
        return float(np.linalg.norm(tl - hl)), float(np.linalg.norm(tr - hr))

    def _set_kinematic(self, on):
        from pxr import UsdPhysics
        UsdPhysics.RigidBodyAPI(self.box_prim).GetKinematicEnabledAttr().Set(bool(on))

    def close(self):
        """Returns (grasped, errors)."""
        self.closed = True
        el, er = self.handle_errors()
        if el < GRASP_TOL and er < GRASP_TOL:
            (tl, tr), wp, wq = self.tcps()
            # held pose: handle centres on the gripper centres, box upright
            y = (tl - tr) / np.linalg.norm(tl - tr)
            x = np.cross(y, [0.0, 0.0, 1.0]); x /= np.linalg.norm(x)
            R = np.stack([x, y, np.cross(x, y)], axis=1)
            p = 0.5 * (tl + tr) - R @ np.array([0.0, 0.0, HANDLE_Z])
            Rw = _quat_to_R(wq[0])
            self.offset = (Rw.T @ (p - wp[0]), Rw.T @ R)
            self.blend = (self.box_pose(), int(GRASP_BLEND_S / 0.002))
            self._set_kinematic(True)
            self.held = True
        return self.held, (el, er)

    def open(self):
        self.closed = False
        if self.held:
            self.held = False
            self._set_kinematic(False)
            self.box.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))

    def step(self):
        """Every physics step while held: the box follows the left hand."""
        if not self.held:
            return
        _, wp, wq = self.tcps()
        Rw = _quat_to_R(wq[0])
        p, R = wp[0] + Rw @ self.offset[0], Rw @ self.offset[1]
        if self.blend is not None:
            (p0, R0), left = self.blend
            n = int(GRASP_BLEND_S / 0.002)
            u = 1.0 - left / n
            u = 0.5 - 0.5 * math.cos(math.pi * u)
            from scipy.spatial.transform import Rotation, Slerp
            R = Slerp([0, 1], Rotation.from_matrix([R0, R]))([u]).as_matrix()[0]
            p = (1 - u) * p0 + u * p
            self.blend = ((p0, R0), left - 1) if left > 1 else None
        self.box.set_world_poses(positions=[p.tolist()], orientations=[_R_to_quat(R)])

    def reset(self):
        """Put the box back on the pickup table."""
        self.open()
        bx, by, byaw = BOX_PICK
        self.box.set_world_poses(positions=[[bx, by, PICK_TABLE["top"] + BOX_HALF_H + 0.002]],
                                 orientations=[[math.cos(byaw / 2), 0.0, 0.0, math.sin(byaw / 2)]])
        self.box.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
