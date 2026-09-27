#!/usr/bin/env python
"""Generate an Isaac-Sim-ready URDF of the EXACT robot the policy was trained on.

  python deploy/isaac/make_isaac_urdf.py            # -> deploy/isaac/x1_isaac/

Source of truth is the compiled evaluation model (assets/x1_full.xml), not the
vendor URDF, because the vendor URDF differs from what the policy saw:
  - its arm joints are `fixed`   -> here: 12 revolute arm joints (same axes)
  - ankle effort 80 Nm           -> here: 18 Nm (the trained limit)
  - mesh foot collision          -> here: the vendor MJCF 4-point sole
                                    (2 mm spheres), exactly as trained
  - no joint damping             -> here: <dynamics damping> as trained
Body frames, inertials, joint axes/ranges and torque limits are read from the
COMPILED MuJoCo model; visual-mesh placements from the XML (MuJoCo re-centres
meshes when compiling, so compiled geom poses would misplace the STL files).

URDF cannot carry rotor armature: it is published in the policy sidecar
(`robot_model.armature`) and set by deploy/isaac/run_x1_policy.py.
"""

import os
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "assets", "x1_full.xml")
OUT_DIR = os.path.join(HERE, "x1_isaac")


def quat_to_rpy(q):
    """MuJoCo quaternion (w, x, y, z) -> URDF rpy (R = Rz(y) Ry(p) Rx(r))."""
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def euler_xyz_to_quat(e):
    """MJCF `euler` with eulerseq="XYZ" (intrinsic) -> quaternion (w,x,y,z)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for i, a in enumerate(e):
        axis = np.zeros(3)
        axis[i] = 1.0
        h = np.array([np.cos(a / 2), *(np.sin(a / 2) * axis)])
        q = quat_mul(q, h)
    return q


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def fmt(v, n=6):
    return " ".join(f"{float(x):.{n}g}" for x in v)


def origin(parent, xyz, rpy):
    ET.SubElement(parent, "origin", xyz=fmt(xyz), rpy=fmt(rpy))


def xml_visuals(src_path):
    """body name -> [(mesh file, pos, quat)] for visual mesh geoms, from XML."""
    root = ET.parse(src_path).getroot()
    meshdir = root.find("compiler").get("meshdir", "")
    files = {m.get("name"): m.get("file") for m in root.find("asset").iter("mesh")}
    out = {}
    for body in root.iter("body"):
        for g in body.findall("geom"):
            if g.get("type") != "mesh" or g.get("class", "").startswith("collision"):
                continue
            pos = np.array([float(v) for v in g.get("pos", "0 0 0").split()])
            if g.get("quat"):
                q = np.array([float(v) for v in g.get("quat").split()])
            elif g.get("euler"):
                q = euler_xyz_to_quat([float(v) for v in g.get("euler").split()])
            else:
                q = np.array([1.0, 0.0, 0.0, 0.0])
            out.setdefault(body.get("name"), []).append(
                (os.path.normpath(os.path.join(os.path.dirname(src_path), meshdir,
                                               files[g.get("mesh")])), pos, q))
    return out


def build(src=SRC, out_dir=OUT_DIR):
    m = mujoco.MjModel.from_xml_path(src)
    visuals = xml_visuals(src)
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)

    robot = ET.Element("robot", name="agibot_x1_policy")
    body_name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    jnt_of_body = {m.jnt_bodyid[j]: j for j in range(m.njnt)
                   if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
    act_of_jnt = {m.actuator_trnid[a, 0]: a for a in range(m.nu)}
    copied = set()

    for b in range(1, m.nbody):
        link = ET.SubElement(robot, "link", name=body_name(b))
        inert = ET.SubElement(link, "inertial")
        origin(inert, m.body_ipos[b], quat_to_rpy(m.body_iquat[b]))
        ET.SubElement(inert, "mass", value=f"{m.body_mass[b]:.6g}")
        I = m.body_inertia[b]
        ET.SubElement(inert, "inertia", ixx=f"{I[0]:.6g}", iyy=f"{I[1]:.6g}",
                      izz=f"{I[2]:.6g}", ixy="0", ixz="0", iyz="0")
        for path, pos, q in visuals.get(body_name(b), []):
            fname = os.path.basename(path)
            if fname not in copied:
                shutil.copy(path, os.path.join(out_dir, "meshes", fname))
                copied.add(fname)
            vis = ET.SubElement(link, "visual")
            origin(vis, pos, quat_to_rpy(q))
            geo = ET.SubElement(vis, "geometry")
            ET.SubElement(geo, "mesh", filename=f"meshes/{fname}")
        # collision: ONLY the trained sole contact spheres
        for g in range(m.ngeom):
            if (m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                    and m.geom_contype[g] and "ankle_roll" in body_name(b)):
                col = ET.SubElement(link, "collision", name=f"{body_name(b)}_sole_{g}")
                origin(col, m.geom_pos[g], (0, 0, 0))
                geo = ET.SubElement(col, "geometry")
                ET.SubElement(geo, "sphere", radius=f"{m.geom_size[g][0]:.6g}")

        if b == 1:
            continue  # pelvis: floating-base root link, no joint
        parent = body_name(m.body_parentid[b])
        j = jnt_of_body.get(b)
        if j is None:
            joint = ET.SubElement(robot, "joint", name=f"{body_name(b)}_fixed", type="fixed")
        else:
            if np.linalg.norm(m.jnt_pos[j]) > 1e-9:
                sys.exit(f"joint {j} not at its body origin; converter assumes it is")
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            joint = ET.SubElement(robot, "joint", name=jname, type="revolute")
            ET.SubElement(joint, "axis", xyz=fmt(m.jnt_axis[j]))
            a = act_of_jnt[j]
            ET.SubElement(joint, "limit", lower=f"{m.jnt_range[j][0]:.6g}",
                          upper=f"{m.jnt_range[j][1]:.6g}",
                          effort=f"{m.actuator_ctrlrange[a][1]:.6g}", velocity="30")
            d = m.jnt_dofadr[j]
            ET.SubElement(joint, "dynamics", damping=f"{m.dof_damping[d]:.6g}",
                          friction=f"{m.dof_frictionloss[d]:.6g}")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=body_name(b))
        origin(joint, m.body_pos[b], quat_to_rpy(m.body_quat[b]))

    ET.indent(robot, space="  ")
    path = os.path.join(out_dir, "x1_isaac.urdf")
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<!-- GENERATED by deploy/isaac/make_isaac_urdf.py '
                'from assets/x1_full.xml: the exact robot the policy was trained on. '
                'Do not hand-edit. -->\n' + ET.tostring(robot, encoding="unicode") + "\n")
    return path, len(copied)


def verify(urdf_path, src=SRC):
    """Load the URDF back into MuJoCo and compare to the trained model."""
    txt = open(urdf_path).read().replace(
        '<robot name="agibot_x1_policy">',
        '<robot name="agibot_x1_policy"><mujoco><compiler meshdir="./" '
        'discardvisual="false" fusestatic="false"/></mujoco>', 1)
    tmp = os.path.join(os.path.dirname(urdf_path), "_verify.urdf")
    open(tmp, "w").write(txt)
    try:
        u = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    t = mujoco.MjModel.from_xml_path(src)
    ju = [u.joint(i).name for i in range(u.njnt)]
    jt = [t.joint(i).name for i in range(1, t.njnt)]   # trained has a free joint
    assert ju == jt, f"joint order differs:\n{ju}\n{jt}"
    mu, mt = u.body_subtreemass[0], t.body_subtreemass[1]
    du, dt_ = mujoco.MjData(u), mujoco.MjData(t)
    pose = t.keyframe("home").qpos
    dt_.qpos[:] = pose
    dt_.qpos[:3] = 0.0            # pelvis at the origin, like the fixed URDF root
    dt_.qpos[3:7] = [1, 0, 0, 0]
    du.qpos[:] = pose[7:]
    mujoco.mj_kinematics(u, du)
    mujoco.mj_kinematics(t, dt_)
    worst = 0.0
    for b in range(1, t.nbody):
        name = t.body(b).name
        worst = max(worst, float(np.linalg.norm(du.xpos[u.body(name).id] - dt_.xpos[b])))
    assert np.allclose(u.jnt_axis, t.jnt_axis[1:], atol=1e-6)
    assert np.allclose(u.jnt_range, t.jnt_range[1:], atol=1e-5)
    print(f"verify: {len(ju)} joints in trained order; mass {mu:.3f} vs {mt:.3f} kg; "
          f"max body position error at the home pose {worst*1000:.4f} mm")
    return worst


if __name__ == "__main__":
    path, n = build()
    print(f"URDF -> {path} ({n} meshes copied)")
    err = verify(path)
    if err > 1e-4:
        sys.exit(f"FAIL: kinematics differ by {err*1000:.3f} mm")
