"""Shared helpers for the box-squeeze feasibility tests (manip/*.py)."""

import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SCENE = os.path.join(HERE, "models", "x1_box_scene.xml")
SCENE_FIXED = os.path.join(HERE, "models", "x1_box_scene_fixed.xml")
SIDECAR = os.path.join(REPO, "policy", "student.json")
ONNX = os.path.join(REPO, "policy", "student.onnx")
VIDEO_DIR = os.path.join(REPO, "outputs", "videos", "manip")

PELVIS_Z = 0.613
BOX_HALF = np.array([0.26, 0.255, 0.26])
BOX_MASS = 2.0
G = 9.81

with open(SIDECAR) as _f:
    SC = json.load(_f)
JOINTS = SC["robot_model"]["joint_order"]
DEFAULT_POSE = np.asarray(SC["default_joint_angles"], float)
KP = np.asarray(SC["pd_gains"]["kp"], float)
KD = np.asarray(SC["pd_gains"]["kd"], float)
TAU_MAX = np.asarray(SC["robot_model"]["torque_limits_nm"], float)
NOISE_STD = SC["training_obs_noise_std"]
N_LEG = 12
ARM = slice(12, 24)
ARM_NAMES = [n.replace("_joint", "") for n in JOINTS[12:18]]


class Scene:
    """Index bookkeeping for either scene (fixed or floating base)."""

    def __init__(self, fixed=True, xml=None):
        """xml: another scene with the same naming (box/box_free/box_geom,
        table); default the squeeze scenes."""
        self.fixed = fixed
        self.m = mujoco.MjModel.from_xml_path(xml or (SCENE_FIXED if fixed else SCENE))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINTS])
        self.act = self._actuators()
        bj = m.joint("box_free")
        self.box_q = m.jnt_qposadr[bj.id]
        self.box_v = m.jnt_dofadr[bj.id]
        self.box_bid = m.body("box").id
        self.box_gid = m.geom("box_geom").id
        self.table_gid = m.geom("table").id
        self.table_mocap = m.body_mocapid[m.body("table").id]
        if m.nsite and any(m.site(i).name == "left_pad_site" for i in range(m.nsite)):
            self.pad_site = [m.site("left_pad_site").id, m.site("right_pad_site").id]
            self.pad_gid = [m.geom("left_pad").id, m.geom("right_pad").id]
            self.tip_site = [m.site("left_tip_site").id, m.site("right_tip_site").id]
        self.arm_gids = [g for g in range(m.ngeom) if m.geom_contype[g] == 16]
        self.torso_bid = m.body("torso_link").id
        self.pelvis_bid = m.body("pelvis").id
        self.arm_jnt_lo = m.jnt_range[[m.joint(n).id for n in JOINTS[12:]], 0]
        self.arm_jnt_hi = m.jnt_range[[m.joint(n).id for n in JOINTS[12:]], 1]
        m.actuator_ctrlrange[self.act, 0] = -TAU_MAX
        m.actuator_ctrlrange[self.act, 1] = TAU_MAX
        self.reset()

    def _actuators(self):
        m = self.m
        out = []
        for n in JOINTS:
            jid = m.joint(n).id
            out.append(int(np.where((m.actuator_trnid[:, 0] == jid)
                                    & (m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))[0][0]))
        return np.array(out)

    # ------------------------------------------------------------- state
    def reset(self, arm=None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        if not self.fixed:
            d.qpos[0:3] = [0, 0, PELVIS_Z]
            d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = DEFAULT_POSE
        if arm is not None:
            d.qpos[self.qadr[ARM]] = arm
        self.park_box()
        self.set_table(None)
        mujoco.mj_forward(m, d)

    def q(self):
        return self.d.qpos[self.qadr].copy()

    def qd(self):
        return self.d.qvel[self.dadr].copy()

    def set_arm(self, arm):
        self.d.qpos[self.qadr[ARM]] = arm
        mujoco.mj_kinematics(self.m, self.d)

    def park_box(self):
        """Move the box out of the way (far away, resting on the floor)."""
        self.place_box([5.0, 0.0, BOX_HALF[2]])

    def place_box(self, centre, quat=(1, 0, 0, 0)):
        d = self.d
        d.qpos[self.box_q:self.box_q + 3] = centre
        d.qpos[self.box_q + 3:self.box_q + 7] = quat
        d.qvel[self.box_v:self.box_v + 6] = 0.0

    def box_pos(self):
        return self.d.qpos[self.box_q:self.box_q + 3].copy()

    def box_tilt_deg(self):
        R = self.d.xmat[self.box_bid].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    def set_table(self, top_z, x_centre=0.6):
        """Table top at top_z (None: move it out of the way)."""
        hz = self.m.geom_size[self.table_gid, 2]
        pos = [5.0, 5.0, -2.0] if top_z is None else [x_centre, 0.0, top_z - hz]
        self.d.mocap_pos[self.table_mocap] = pos

    def set_friction(self, mu, torsion=None):
        """Sliding friction of the arm proxies and the box. torsion (m): if
        set, condim 4 with that torsional coefficient (a contact patch);
        otherwise condim 3 (point contacts, no torsional resistance)."""
        m = self.m
        for g in self.arm_gids + [self.box_gid]:
            m.geom_friction[g, 0] = mu
            m.geom_condim[g] = 3 if torsion is None else 4
            m.geom_friction[g, 1] = 0.0 if torsion is None else torsion

    # ------------------------------------------------------------- control
    def pd(self, target, arm_gravity_comp=False):
        """Joint PD at the sidecar gains (500 Hz), clipped at the torque limits.
        arm_gravity_comp adds the arm links' own gravity/bias torque
        (qfrc_bias) to the arm joints; the box load is NOT compensated."""
        tgt = np.asarray(target, float)
        tau = KP * (tgt - self.q()) - KD * self.qd()
        if arm_gravity_comp:
            tau[ARM] += self.d.qfrc_bias[self.dadr[ARM]]
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        self.d.ctrl[self.act] = tau
        return tau

    # ------------------------------------------------------------- contacts
    def box_contacts(self):
        """Per-side normal force (N) on the box from the arm proxies, plus the
        table/torso contact forces. Returns dict."""
        m, d = self.m, self.d
        out = {"left": 0.0, "right": 0.0, "left_pad": 0.0, "right_pad": 0.0,
               "torso": 0.0, "table": 0.0, "floor": 0.0}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self.box_gid not in (g1, g2):
                continue
            other = g2 if g1 == self.box_gid else g1
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(f6[0])
            name = m.geom(other).name or ""
            bname = m.body(m.geom_bodyid[other]).name
            if other == self.table_gid:
                out["table"] += fn
            elif name == "floor":
                out["floor"] += fn
            elif m.geom_contype[other] == 16:
                side = "left" if bname.startswith("left") else "right"
                out[side] += fn
                if other in self.pad_gid:
                    out[side + "_pad"] += fn
            else:
                out["torso"] += fn
        return out


def quat_rotate_inv(q, v):
    w, u = q[0], q[1:4]
    return v + 2.0 * np.cross(-u, np.cross(-u, v) + w * v)


def renderer(scene, w=960, h=540):
    r = mujoco.Renderer(scene.m, h, w)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.25, 0.0, 0.85]
    cam.distance, cam.elevation, cam.azimuth = 2.3, -15.0, 145.0
    return r, opt, cam


def snap(scene, r, opt, cam):
    r.update_scene(scene.d, cam, opt)
    return r.render()
