#!/usr/bin/env python
"""Generate assets/x1_full.xml (eval) and assets/x1_mjx.xml (training) from
the vendor AgiBot X1 MJCF (assets/x1_vendor/mjcf/xyber_x1_serial.xml, the
model the vendor's own sim2sim uses). Mechanical and repeatable: re-run after
changing configs/actuators.yaml default_pose, then `pytest tests/test_model.py`.

Both outputs:
  - bodies renamed for the env contract: x1-body -> pelvis, body_pitch ->
    torso_link. Joint names are the vendor's.
  - 12 leg joints + 12 ARM joints (6 per arm, axes from the vendor URDF,
    see ARM_JOINTS; the waist stays fixed). qpos order = actuator order:
    left leg 6, right leg 6, left arm 6, right arm 6.
  - sensors and the vendor keyframe stripped; timestep 0.002 (500 Hz PD).
  - joint armature 0 -> ARMATURE (see below); vendor damping 1.0 kept.
  - floor + light; left/right_payload bodies (0.05 kg) at the wrist-roll
    bodies, used as the hand point (no payload is sampled for stand/walk).
  - `home` keyframe = configs/actuators.yaml default_pose with the pelvis
    height set so the soles touch the floor.

x1_full.xml keeps the vendor collision set: base and torso meshes plus four
r=2 mm sole spheres per foot. x1_mjx.xml keeps the SAME four sole spheres per
foot (the only foot-floor contact), drops the meshes, and adds: a
non-colliding reference box per sole (pose/support polygon), one foot-self
sphere per foot, one hand sphere per hand (floor only), one torso capsule.
Bitmask: 1 floor | 2 feet | 4 hands | 8 torso | 16 foot-self.
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ASSETS = os.path.join(REPO, "assets")
SRC = os.path.join(ASSETS, "x1_vendor", "mjcf", "xyber_x1_serial.xml")
MESHDIR = os.path.join(ASSETS, "x1_vendor", "meshes")
RENAME = {"x1-body": "pelvis", "body_pitch": "torso_link"}

# The vendor MJCF has zero armature. At 500 Hz with the vendor's stiff knee
# (kp 100) a zero-armature PD hold spikes to 10 rad/s joint speed; 0.01 brings
# it to the 1 kHz behaviour. The X1 training randomised armature over
# [0.0001, 0.05] (X1 cfg domain_rand.joint_armature_range); 0.01 sits inside.
ARMATURE = 0.01

# Hardware added for the office tasks (deploy/isaac/sensors/x1_sensors.yaml):
# OmniPicker gripper 0.43 kg + RealSense D405 0.06 kg per wrist, Livox Mid-360
# 0.265 kg on top of the head (torso_link frame: y is up).
GRIPPER_MASS = 0.49
MID360_MASS = 0.265
MID360_POS = [0.0345, 0.51, 0.0]
SOLE_HALF_THICKNESS = 0.01

# Arm joints. The vendor model welds the arms; the vendor URDF keeps these six
# joints per arm as `fixed` but still records their axes, which are expressed
# in the same (child-link) frames the MJCF bodies use (checked on the legs:
# every leg axis matches between the two files). Axes below are copied from
# x1.urdf. Ranges are NOT in either vendor file (URDF says +-3.14); they are
# chosen to cover the task poses (forward, sideways, overhead, back, carry)
# without the arm passing through the torso. Order per arm = qpos order.
#   joint (MJCF body)        axis       range (rad)   torque (Nm, assumed)
ARM_JOINTS = {
    "left": [("left_shoulder_pitch", "0 0 -1", (-3.1, 0.8), 40),
             ("left_shoudler_roll", "0 0 -1", (-1.8, 0.3), 40),   # vendor typo kept
             ("left_shoulder_yaw", "0 0 1", (-1.6, 1.6), 20),
             ("left_elbow_pitch", "0 0 1", (0.0, 2.2), 20),
             ("left_elbow_yaw", "0 0 1", (-1.6, 1.6), 10),
             ("left_wrist_pitch", "-1 0 0", (-0.8, 0.8), 6)],
    "right": [("right_shoulder_pitch", "0 0 1", (-3.1, 0.8), 40),
              ("right_shoulder_roll", "0 0 1", (-1.8, 0.3), 40),
              ("right_shoulder_yaw", "0 0 1", (-1.6, 1.6), 20),
              ("right_elbow_pitch", "0 0 -1", (0.0, 2.2), 20),
              ("right_elbow_yaw", "0 0 -1", (-1.6, 1.6), 10),
              ("right_wrist_pitch", "1 0 0", (-0.8, 0.8), 6)],
}


def arm_joint_name(body):
    """Joint names normalise the vendor's 'shoudler' typo."""
    return body.replace("shoudler", "shoulder") + "_joint"


def _add_arm_joints(root):
    """Hinge joints + motors for both arms, and move the torso subtree after
    the legs so qpos order = actuator order = legs 12 | left arm 6 | right
    arm 6 (the vendor tree lists the torso first)."""
    pelvis = _find_body(root, "pelvis")
    waist = _find_body(root, "body_yaw")
    pelvis.remove(waist)
    pelvis.append(waist)
    act = root.find("actuator")
    for side in ("left", "right"):
        for body, axis, (lo, hi), tau in ARM_JOINTS[side]:
            name = arm_joint_name(body)
            _find_body(root, body).insert(0, ET.Element(
                "joint", name=name, type="hinge", axis=axis, range=f"{lo} {hi}",
                damping="0.5"))
            ET.SubElement(act, "motor", name="motor_" + name, joint=name,
                          ctrlrange=f"{-tau} {tau}")


def _find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


def _base_tree():
    tree = ET.parse(SRC)
    root = tree.getroot()
    root.find("compiler").set("meshdir", os.path.relpath(MESHDIR, ASSETS) + "/")
    for b in root.iter("body"):
        if b.get("name") in RENAME:
            b.set("name", RENAME[b.get("name")])
    for tag in ("sensor", "keyframe"):
        el = root.find(tag)
        if el is not None:
            root.remove(el)
    default = root.find("default")
    ET.SubElement(default, "joint", armature=str(ARMATURE))
    _add_arm_joints(root)
    for side in ("left", "right"):
        wrist = _find_body(root, f"{side}_wrist_roll")
        pb = ET.SubElement(wrist, "body", name=f"{side}_payload", pos="0 0 0")
        ET.SubElement(pb, "inertial", pos="0 0 0", mass="0.05",
                      diaginertia="2e-4 2e-4 2e-4")
        ET.SubElement(pb, "geom", name=f"{side}_payload_viz", type="sphere",
                      size="0.03", contype="0", conaffinity="0", group="1",
                      rgba="0.9 0.4 0.1 0.5")
        # OmniPicker gripper (0.43 kg) + wrist D405 (0.06 kg) as one rigid
        # mass on its own body, so payload DR (which rewrites the payload
        # body's mass) cannot erase it. Centred 0.09 m along the approach
        # direction (-y of the wrist frame); see deploy/isaac/sensors/x1_sensors.yaml.
        gb = ET.SubElement(wrist, "body", name=f"{side}_gripper", pos="0 -0.09 0")
        ET.SubElement(gb, "inertial", pos="0 0 0", mass=str(GRIPPER_MASS),
                      diaginertia="6e-4 3e-4 6e-4")
        ET.SubElement(gb, "geom", name=f"{side}_gripper_viz", type="box",
                      size="0.04 0.06 0.03", contype="0", conaffinity="0",
                      group="1", rgba="0.15 0.15 0.15 1")
    # Livox Mid-360 upside down on top of the head (0.265 kg), torso_link frame
    torso = _find_body(root, "torso_link")
    lb = ET.SubElement(torso, "body", name="mid360", pos=_fmt(MID360_POS))
    ET.SubElement(lb, "inertial", pos="0 0 0", mass=str(MID360_MASS),
                  diaginertia="2e-4 2e-4 2e-4")
    ET.SubElement(lb, "geom", name="mid360_viz", type="cylinder",
                  size="0.0325 0.03", zaxis="0 1 0", contype="0",
                  conaffinity="0", group="1", rgba="0.1 0.1 0.1 1")
    return tree, root


def _add_scene(root, mjx: bool):
    opt = root.find("option")
    opt.set("timestep", "0.002")
    if mjx:
        opt.set("iterations", "2")
        opt.set("ls_iterations", "4")
        opt.set("solver", "Newton")
        opt.set("cone", "pyramidal")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", type="2d", name="groundplane", builtin="checker",
                  mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
                  markrgb="0.8 0.8 0.8", width="300", height="300")
    ET.SubElement(asset, "material", name="groundplane", texture="groundplane",
                  texuniform="true", texrepeat="5 5", reflectance="0.2")
    wb = root.find("worldbody")
    # full model: vendor collision geoms are contype 1, floor conaffinity 7 as
    # in the vendor environment/flat.xml. mjx model: see bitmask above.
    wb.insert(0, ET.Element("geom", name="floor", size="0 0 0.05", type="plane",
                            material="groundplane", contype="1",
                            conaffinity="1" if mjx else "7", condim="3",
                            friction="1 0.02 0.01"))
    wb.insert(0, ET.Element("light", pos="0 0 3.5", dir="0 0 -1", directional="true"))


def _posed(xml_text, pose):
    import mujoco
    cwd = os.getcwd()
    os.chdir(ASSETS)
    try:
        m = mujoco.MjModel.from_xml_string(xml_text)
    finally:
        os.chdir(cwd)
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[2] = 1.0
    d.qpos[3] = 1.0
    d.qpos[7:] = pose
    mujoco.mj_forward(m, d)
    return m, d


def _local(d, bid, p_world):
    R = d.xmat[bid].reshape(3, 3)
    return R.T @ (np.asarray(p_world) - d.xpos[bid])


def _world_aligned_quat(d, bid):
    """Quaternion (in the body frame) of a frame aligned with the world at
    the posed configuration, so a box's local z is vertical at `home` (the
    env's support-polygon code reads size[:2] as the ground-plane extents)."""
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, d.xmat[bid].reshape(3, 3).T.ravel())
    return q


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _primitive_collisions(root, pose):
    import mujoco
    # measure the vendor sole points before removing them
    m, d = _posed(ET.tostring(root, encoding="unicode"), pose)
    feet = {}
    for side in ("left", "right"):
        bid = m.body(f"{side}_ankle_roll_link").id
        pts = [g for g in range(m.ngeom)
               if m.geom_bodyid[g] == bid and m.geom_contype[g]
               and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        xyz = np.array([d.geom_xpos[g] for g in pts])
        r = m.geom_size[pts[0]][0]
        half = [(xyz[:, 0].max() - xyz[:, 0].min()) / 2 + r,
                (xyz[:, 1].max() - xyz[:, 1].min()) / 2 + r,
                SOLE_HALF_THICKNESS]
        sole_z = xyz[:, 2].min() - r
        centre = [xyz[:, 0].mean(), xyz[:, 1].mean(), sole_z + SOLE_HALF_THICKNESS]
        feet[side] = dict(bid=bid, half=half, centre=centre,
                          quat=_world_aligned_quat(d, bid))
    torso_id = m.body("torso_link").id
    torso_base = d.xpos[torso_id].copy()

    sole_bodies = {f"{side}_ankle_roll_link": side for side in ("left", "right")}
    for body in root.iter("body"):
        side = sole_bodies.get(body.get("name"))
        k = 0
        for g in list(body.findall("geom")):
            if not g.get("class", "").startswith("collision"):
                continue
            if side is not None and g.get("type") == "sphere":
                # KEEP the vendor sole spheres as THE foot contact, identical
                # to x1_full.xml. teacher_v9 trained on a box sole and lost
                # most of its walking on the vendor 4-point sole (backward
                # walk 83 % on the box model vs 0 % on the full model).
                g.attrib.pop("class")
                g.set("name", f"{side}_sole_{k}")
                g.set("contype", "2")
                g.set("conaffinity", "1")
                g.set("group", "3")
                g.set("friction", "1 0.02 0.01")
                k += 1
            else:
                body.remove(g)
    for side, f in feet.items():
        foot = _find_body(root, f"{side}_ankle_roll_link")
        # NON-colliding reference box over the sole: the env reads its pose
        # and half-extents for foot position and the support polygon. Contact
        # is made by the vendor sole spheres kept above.
        ET.SubElement(foot, "geom", name=f"{side}_foot", type="box",
                      size=_fmt(f["half"]),
                      pos=_fmt(_local(d, f["bid"], f["centre"])),
                      quat=_fmt(f["quat"]), contype="0", conaffinity="0",
                      group="4", rgba="0.7 0.3 0.3 0.2")
        up = np.asarray(f["centre"]) + [0.0, 0.0, 0.03]
        ET.SubElement(foot, "geom", name=f"{side}_foot_self", type="sphere",
                      size="0.04", pos=_fmt(_local(d, f["bid"], up)),
                      contype="16", conaffinity="16", group="3",
                      rgba="0.7 0.3 0.3 0.4")
        hand = _find_body(root, f"{side}_wrist_roll")
        # floor only: the arms are task-commanded, so arm-torso contact is
        # not something the leg policy should be terminated for.
        ET.SubElement(hand, "geom", name=f"{side}_hand", type="sphere",
                      size="0.04", pos="0 0 0", contype="4", conaffinity="1",
                      group="3", rgba="0.7 0.3 0.3 0.4")
    torso = _find_body(root, "torso_link")
    a = _local(d, torso_id, torso_base + [0.0, 0.0, 0.08])
    b = _local(d, torso_id, torso_base + [0.0, 0.0, 0.33])
    ET.SubElement(torso, "geom", name="torso_capsule", type="capsule",
                  size="0.10", fromto=_fmt(np.concatenate([a, b])), contype="8",
                  conaffinity="0", group="3", rgba="0.7 0.3 0.3 0.4")
    # conaffinity 0: its only partner used to be the hands, which are now
    # task-commanded and floor-only. Kept (inert) as the torso's primitive.


def _standing_height(xml_text, pose):
    """Pelvis z such that the lowest floor-touching point is at z=0."""
    import mujoco
    m, d = _posed(xml_text, pose)
    lowest = np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        if not ((m.geom_contype[g] & 1) or (m.geom_conaffinity[g] & 1)):
            continue
        t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = d.geom_xpos[g][2] - m.geom_size[g][0]
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3)
            z = d.geom_xpos[g][2] - np.abs(R[2]) @ m.geom_size[g]
        else:
            continue  # meshes (base/torso) are far above the floor at home
        lowest = min(lowest, z)
    return 1.0 - lowest


def build(mjx: bool, pose, out):
    tree, root = _base_tree()
    root.set("model", "x1_mjx" if mjx else "x1_full")
    _add_scene(root, mjx)
    if mjx:
        _primitive_collisions(root, pose)
    h = _standing_height(ET.tostring(root, encoding="unicode"), pose)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="home",
                  qpos=_fmt([0.0, 0.0, h, 1.0, 0.0, 0.0, 0.0] + list(pose)))
    ET.indent(tree, space="  ")
    header = ("<!-- GENERATED by scripts/build_assets.py from "
              "assets/x1_vendor/mjcf/xyber_x1_serial.xml. Do not hand-edit; "
              "see assets/PROVENANCE.md. -->\n")
    with open(out, "w") as f:
        f.write(header + ET.tostring(root, encoding="unicode") + "\n")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actuators", default=os.path.join(REPO, "configs", "actuators.yaml"))
    args = ap.parse_args()
    with open(args.actuators) as f:
        pose = yaml.safe_load(f)["default_pose"]
    h_full = build(False, pose, os.path.join(ASSETS, "x1_full.xml"))
    h_mjx = build(True, pose, os.path.join(ASSETS, "x1_mjx.xml"))
    print(f"x1_full.xml home pelvis z = {h_full:.4f} m")
    print(f"x1_mjx.xml  home pelvis z = {h_mjx:.4f} m")


if __name__ == "__main__":
    main()
