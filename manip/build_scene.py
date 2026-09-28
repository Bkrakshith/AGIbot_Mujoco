#!/usr/bin/env python
"""Build the box-squeeze test scenes from assets/x1_full.xml (read only).

  python manip/build_scene.py

Writes manip/models/x1_box_scene.xml (floating base, for the walking policy)
and manip/models/x1_box_scene_fixed.xml (pelvis fixed at 0.613 m, for the
kinematic and static-hold tests). Both add:

  - a closed-gripper pad per hand (OmniPicker, 0.43 kg added to the payload
    body): box 0.08 x 0.12 x 0.06 m on the wrist_roll body; a site on that face
    ("<side>_pad_site", z axis = pad normal pointing at the box) and one on
    its distal-medial edge ("<side>_tip_site", the closed fingertips);
  - capsules on the upper arm and forearm, fitted to the vendor meshes;
  - the box (0.51 w x 0.52 d x 0.52 h, 2.0 kg, free joint);
  - a table (mocap body; height set at run time, see common.Scene.set_table).

Contact bits: arm proxies 16, box 32, table 1 (affinity 16 so the pads hit it),
vendor robot geoms 1. The box collides with the floor, table, torso/pelvis
meshes and the arm proxies. Arm proxies do not collide with the robot itself.
Proxy geoms are massless: the bodies keep their vendor inertials; the gripper
mass goes on the payload body at the pad centre.
"""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "assets", "x1_full.xml")
OUT_DIR = os.path.join(HERE, "models")

BOX_HALF = (0.26, 0.255, 0.26)      # x (depth, 0.52), y (width, 0.51), z (0.52)
BOX_MASS = 2.0
FRICTION = 0.5                       # pads, forearms and box; changed at run time
# pad in the wrist_roll frame: y = distal (down the hand), z = medial
# Closed OmniPicker gripper (1-DoF pinch, 0.43 kg) on the payload frame,
# approach along -y, flange ~0.03 m out: palm ~80 x 60 x 90 mm plus closed
# fingers ~30 mm -> y from -0.03 to -0.15, centred on the wrist axis. The
# vendor wrist_roll STL already reaches y = -0.104, so the pad overlaps it
# (no double length).
PAD_HALF = (0.04, 0.06, 0.03)
PAD_POS = (0.0, -0.09, 0.0)          # medial face at z = +0.03
GRIPPER_MASS = 0.43
TABLE_HALF = (0.3, 0.35, 0.6)        # top at mocap z + 0.6
ARM_CAPSULE_BODIES = ("shoulder_yaw", "elbow_pitch", "elbow_yaw")


def _fmt(v):
    return " ".join(f"{x:.5g}" for x in v)


def _find(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


def _fit_capsule(m, body_name):
    """Capsule along the principal axis of the body's visual mesh (body frame)."""
    b = m.body(body_name).id
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.append(v @ R.reshape(3, 3).T + m.geom_pos[g])
    p = np.concatenate(pts)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    ax = vt[0]
    s = (p - c) @ ax
    radial = np.linalg.norm((p - c) - np.outer(s, ax), axis=1)
    r = float(np.percentile(radial, 90))
    lo, hi = s.min() + r, s.max() - r
    if hi <= lo:
        lo = hi = 0.5 * (s.min() + s.max())
    return np.concatenate([c + lo * ax, c + hi * ax]), r


def build(fixed_base: bool, out):
    tree = ET.parse(SRC)
    root = tree.getroot()
    root.set("model", "x1_box_scene_fixed" if fixed_base else "x1_box_scene")
    meshdir = os.path.relpath(os.path.join(REPO, "assets", "x1_vendor", "meshes"), OUT_DIR)
    root.find("compiler").set("meshdir", meshdir + "/")
    ref = mujoco.MjModel.from_xml_path(SRC)

    for side in ("left", "right"):
        wrist = _find(root, f"{side}_wrist_roll")
        pl = _find(root, f"{side}_payload").find("inertial")
        pl.set("mass", f"{0.05 + GRIPPER_MASS:.3f}")
        pl.set("pos", _fmt(PAD_POS))
        pl.set("diaginertia", "1.2e-3 6e-4 1.2e-3")
        ET.SubElement(wrist, "geom", name=f"{side}_pad", type="box", size=_fmt(PAD_HALF),
                      pos=_fmt(PAD_POS), contype="16", conaffinity="0", condim="3",
                      friction=f"{FRICTION} 0.005 0.0001", mass="0", group="1",
                      rgba="0.15 0.15 0.15 1")
        face = (PAD_POS[0], PAD_POS[1], PAD_POS[2] + PAD_HALF[2])
        ET.SubElement(wrist, "site", name=f"{side}_pad_site", pos=_fmt(face), size="0.01",
                      rgba="1 0 0 1", group="1")
        tip = (PAD_POS[0], PAD_POS[1] - PAD_HALF[1], face[2])   # distal-medial edge
        ET.SubElement(wrist, "site", name=f"{side}_tip_site", pos=_fmt(tip), size="0.01",
                      rgba="0 0 1 1", group="1")
        for part in ARM_CAPSULE_BODIES:
            vendor = f"{side}_{part}"
            ft, r = _fit_capsule(ref, vendor)
            ET.SubElement(_find(root, vendor), "geom", name=f"{side}_{part}_cap",
                          type="capsule", fromto=_fmt(ft), size=f"{r:.4f}",
                          contype="16", conaffinity="0", condim="3",
                          friction=f"{FRICTION} 0.005 0.0001", mass="0", group="3",
                          rgba="0.2 0.6 0.9 0.3")

    wb = root.find("worldbody")
    if fixed_base:
        pelvis = _find(root, "pelvis")
        pelvis.remove(pelvis.find("freejoint"))
        pelvis.set("pos", "0 0 0.613")
    # mocap body so its height can move at run time (a world geom cannot:
    # the world body's midphase tree is built at compile time)
    table = ET.SubElement(wb, "body", name="table", mocap="true", pos="0.6 0 0.1")
    ET.SubElement(table, "geom", name="table", type="box", size=_fmt(TABLE_HALF),
                  contype="1", conaffinity="16", condim="3",
                  friction="0.5 0.005 0.0001", rgba="0.55 0.45 0.35 1")
    box = ET.SubElement(wb, "body", name="box", pos="0.6 0 0.9")
    ET.SubElement(box, "freejoint", name="box_free")
    ET.SubElement(box, "geom", name="box_geom", type="box", size=_fmt(BOX_HALF),
                  mass=str(BOX_MASS), contype="32", conaffinity="17", condim="3",
                  friction=f"{FRICTION} 0.005 0.0001", rgba="0.76 0.6 0.42 1")
    ET.SubElement(box, "site", name="box_site", size="0.01")

    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    glob = vis.find("global")
    if glob is None:
        glob = ET.SubElement(vis, "global")
    glob.set("offwidth", "1280")
    glob.set("offheight", "720")

    kf = root.find("keyframe")
    if kf is not None:
        root.remove(kf)   # re-created at run time (qpos layout changed)
    ET.indent(tree, space="  ")
    with open(out, "w") as f:
        f.write("<!-- GENERATED by manip/build_scene.py from assets/x1_full.xml. -->\n"
                + ET.tostring(root, encoding="unicode") + "\n")
    m = mujoco.MjModel.from_xml_path(out)
    return m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fixed, name in ((False, "x1_box_scene.xml"), (True, "x1_box_scene_fixed.xml")):
        m = build(fixed, os.path.join(OUT_DIR, name))
        robot = m.body_subtreemass[m.body("pelvis").id]
        print(f"{name}: nq={m.nq} nu={m.nu} robot mass={robot:.3f} kg "
              f"box={m.body_mass[m.body('box').id]:.2f} kg")


if __name__ == "__main__":
    main()
