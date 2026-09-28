#!/usr/bin/env python
"""Scenes for the handle-box task, generated from assets/x1_full.xml (read
only) and, when it exists, manip/handle_box.yaml (the box design).

Each wrist gets an OmniPicker stand-in (deploy/isaac/sensors/x1_sensors.yaml
gripper.proposed_geometry), mounted on the <side>_payload frame with the
approach along -y:
  - palm 80 (across the jaws, payload x) x 90 (along the approach) x 60 mm,
    from 0.03 m (flange) to 0.12 m out, 0.35 kg;
  - two mirrored prismatic fingers along payload x, 0..0.06 m each (0.12 m
    stroke), finger boxes 18 (jaw axis) x 70 (approach) x 20 mm, 0.04 kg
    each, pad friction 1.0, position servo with a 30 N force limit;
  - TCP site midway between the fingertips, 0.14 m from the flange.
The vendor wrist_roll STL already reaches ~0.10 m past the wrist; the palm
overlaps it (no extra length). Upper-arm/forearm capsules as in
build_scene.py. Box, handles and table use the contact bits of build_scene.py.

  python manip/build_handle_scene.py
"""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from build_scene import ARM_CAPSULE_BODIES, OUT_DIR, REPO, SRC, _find, _fit_capsule, _fmt

SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handle_box.yaml")
BOX_XML = os.path.join(OUT_DIR, "handle_box.xml")

FLANGE = 0.030
PALM = dict(size=(0.080, 0.090, 0.060), mass=0.35)      # payload x, y (approach), z
FINGER = dict(size=(0.018, 0.070, 0.020), mass=0.04)
STROKE = 0.060                                           # per finger
TCP_FROM_FLANGE = 0.140
FINGER_FORCE = 30.0
FINGER_KP = 3000.0                                       # N/m, saturates at 30 N after 1 cm
PAD_FRICTION = 1.0
# narrow table (top at mocap z + 0.6): the hands pass outside it (|y| > 0.19)
# when the arms swing up from "down" to "carry"
TABLE_HALF = (0.12, 0.10, 0.6)
PLACE_TABLE_HALF = (0.30, 0.35, 0.6)


def add_gripper(root, side):
    payload = _find(root, f"{side}_payload")
    y_palm = -(FLANGE + PALM["size"][1] / 2)
    inert = payload.find("inertial")
    inert.set("mass", f"{0.05 + PALM['mass']:.3f}")
    inert.set("pos", _fmt((0.0, y_palm, 0.0)))
    inert.set("diaginertia", "4.8e-4 3.0e-4 4.2e-4")
    ET.SubElement(payload, "geom", name=f"{side}_palm", type="box",
                  size=_fmt(np.array(PALM["size"]) / 2), pos=_fmt((0.0, y_palm, 0.0)),
                  contype="16", conaffinity="0", mass="0", group="1",
                  friction=f"{PAD_FRICTION} 0.005 0.0001", rgba="0.2 0.2 0.22 1")
    y_tcp = -(FLANGE + TCP_FROM_FLANGE)
    fx, fy, fz = np.array(FINGER["size"]) / 2
    for k, sign in (("a", 1.0), ("b", -1.0)):
        fb = ET.SubElement(payload, "body", name=f"{side}_finger_{k}",
                           pos=_fmt((sign * fx, y_tcp, 0.0)))
        ET.SubElement(fb, "joint", name=f"{side}_finger_{k}", type="slide",
                      axis=_fmt((sign, 0.0, 0.0)), range=f"0 {STROKE}", damping="5",
                      armature="0.001")
        ET.SubElement(fb, "inertial", pos="0 0 0", mass=str(FINGER["mass"]),
                      diaginertia="2e-5 2e-5 2e-5")
        ET.SubElement(fb, "geom", name=f"{side}_finger_{k}_geom", type="box",
                      size=_fmt((fx, fy, fz)), contype="16", conaffinity="0", mass="0",
                      condim="4", friction=f"{PAD_FRICTION} 0.01 0.0001", group="1",
                      rgba="0.1 0.1 0.1 1")
    ET.SubElement(payload, "site", name=f"{side}_tcp", pos=_fmt((0.0, y_tcp, 0.0)),
                  size="0.008", rgba="1 0 0 1", group="1")
    act = root.find("actuator")
    for k in ("a", "b"):
        ET.SubElement(act, "position", name=f"{side}_finger_{k}_servo",
                      joint=f"{side}_finger_{k}", kp=str(FINGER_KP), ctrlrange=f"0 {STROKE}",
                      forcerange=f"{-FINGER_FORCE} {FINGER_FORCE}")


def box_body(spec):
    """<body name="box"> with free joint, corrugated body, stand-offs and
    rubber handles, from the spec (box frame = box centre, axes = world at
    the hold pose)."""
    b = spec["box"]
    body = ET.Element("body", name="box", pos="0 0 0")
    ET.SubElement(body, "freejoint", name="box_free")
    half = np.array(b["size"]) / 2
    ET.SubElement(body, "geom", name="box_geom", type="box", size=_fmt(half),
                  mass=str(b["body_mass"]), contype="32", conaffinity="17", condim="3",
                  friction="0.5 0.005 0.0001", rgba="0.76 0.6 0.42 1")
    h = spec["handles"]
    for side in ("left", "right"):
        hs = h[side]
        c = np.array(hs["centre"])
        a = np.array(hs["axis_out"])
        inner = c - a * h["length"] / 2
        root_pt = np.array(hs["standoff_root"])
        ET.SubElement(body, "geom", name=f"{side}_standoff", type="cylinder",
                      fromto=_fmt(np.concatenate([root_pt, inner])),
                      size=str(h["standoff_radius"]), mass=str(h["standoff_mass"]),
                      contype="32", conaffinity="17", rgba="0.3 0.3 0.3 1")
        ET.SubElement(body, "geom", name=f"{side}_handle", type="cylinder",
                      fromto=_fmt(np.concatenate([inner, c + a * h["length"] / 2])),
                      size=str(h["diameter"] / 2), mass=str(h["mass"]),
                      contype="32", conaffinity="17", condim="4",
                      friction=f"{h['friction']} 0.01 0.0001", rgba="0.05 0.05 0.05 1")
        ET.SubElement(body, "site", name=f"{side}_handle_site", pos=_fmt(c), size="0.006",
                      rgba="0 1 0 1")
    return body


def write_box_mjcf(spec, out=BOX_XML):
    root = ET.Element("mujoco", model="handle_box")
    wb = ET.SubElement(root, "worldbody")
    body = box_body(spec)
    body.set("pos", _fmt((0.0, 0.0, spec["box"]["size"][2] / 2)))
    wb.append(body)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(out, "w") as f:
        f.write("<!-- GENERATED by manip/build_handle_scene.py from manip/handle_box.yaml. "
                "Box frame = box centre; handles in that frame. -->\n"
                + ET.tostring(root, encoding="unicode") + "\n")


def build(fixed_base, out, spec=None):
    tree = ET.parse(SRC)
    root = tree.getroot()
    root.set("model", "x1_handle_scene" + ("_fixed" if fixed_base else ""))
    meshdir = os.path.relpath(os.path.join(REPO, "assets", "x1_vendor", "meshes"), OUT_DIR)
    root.find("compiler").set("meshdir", meshdir + "/")
    ref = mujoco.MjModel.from_xml_path(SRC)
    for side in ("left", "right"):
        add_gripper(root, side)
        for part in ARM_CAPSULE_BODIES:
            vendor = f"{side}_{part}"
            ft, r = _fit_capsule(ref, vendor)
            ET.SubElement(_find(root, vendor), "geom", name=f"{side}_{part}_cap",
                          type="capsule", fromto=_fmt(ft), size=f"{r:.4f}",
                          contype="16", conaffinity="0", mass="0", group="3",
                          rgba="0.2 0.6 0.9 0.3")
    wb = root.find("worldbody")
    if fixed_base:
        pelvis = _find(root, "pelvis")
        pelvis.remove(pelvis.find("freejoint"))
        pelvis.set("pos", "0 0 0.613")
    table = ET.SubElement(wb, "body", name="table", mocap="true", pos="5 5 -2")
    ET.SubElement(table, "geom", name="table", type="box", size=_fmt(TABLE_HALF),
                  contype="1", conaffinity="16", condim="3",
                  friction="0.5 0.005 0.0001", rgba="0.55 0.45 0.35 1")
    # place table: full-size (0.6 x 0.7 m top), used only for placing
    ptable = ET.SubElement(wb, "body", name="place_table", mocap="true", pos="-5 5 -2")
    ET.SubElement(ptable, "geom", name="place_table", type="box",
                  size=_fmt(PLACE_TABLE_HALF), contype="1", conaffinity="16", condim="3",
                  friction="0.5 0.005 0.0001", rgba="0.5 0.42 0.33 1")
    if spec is not None:
        wb.append(box_body(spec))
    else:   # placeholder so the index bookkeeping is the same
        body = ET.SubElement(wb, "body", name="box", pos="5 0 0.05")
        ET.SubElement(body, "freejoint", name="box_free")
        ET.SubElement(body, "geom", name="box_geom", type="box", size="0.05 0.05 0.05",
                      mass="0.1", contype="32", conaffinity="17")
    vis = root.find("visual")
    glob = vis.find("global") if vis.find("global") is not None else ET.SubElement(vis, "global")
    glob.set("offwidth", "1280")
    glob.set("offheight", "720")
    kf = root.find("keyframe")
    if kf is not None:
        root.remove(kf)
    ET.indent(tree, space="  ")
    with open(out, "w") as f:
        f.write("<!-- GENERATED by manip/build_handle_scene.py from assets/x1_full.xml"
                + (" and manip/handle_box.yaml" if spec else "") + ". -->\n"
                + ET.tostring(root, encoding="unicode") + "\n")
    return mujoco.MjModel.from_xml_path(out)


def main():
    spec = None
    if os.path.exists(SPEC):
        with open(SPEC) as f:
            spec = yaml.safe_load(f)
        write_box_mjcf(spec)
    for fixed, name in ((False, "x1_handle_scene.xml"), (True, "x1_handle_scene_fixed.xml")):
        m = build(fixed, os.path.join(OUT_DIR, name), spec)
        print(f"{name}: nq={m.nq} nu={m.nu} robot={m.body_subtreemass[m.body('pelvis').id]:.3f} kg "
              f"box={m.body_subtreemass[m.body('box').id]:.3f} kg")


if __name__ == "__main__":
    main()
