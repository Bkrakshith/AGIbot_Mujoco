"""Asset gates: both MJCF variants load, agree on ordering, and the reduced
model's contact set is what PROVENANCE.md promises."""

import mujoco
import numpy as np
import pytest

from x1_locomotion.config import FULL_XML, MJX_XML, load_config
from x1_locomotion.layout import NU


@pytest.fixture(scope="module")
def models():
    return (mujoco.MjModel.from_xml_path(MJX_XML),
            mujoco.MjModel.from_xml_path(FULL_XML))


def test_dof_layout(models):
    """X1 vendor model: free joint + 12 leg hinges, 12 motors."""
    for m in models:
        assert (m.nq, m.nv, m.nu) == (7 + NU, 6 + NU, NU) == (19, 18, 12)


def test_only_leg_joints(models):
    """Waist and arms are fixed links in the vendor X1 model."""
    legs = [f"{s}_{j}" for s in ("left", "right") for j in
            ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch",
             "ankle_roll")]
    for m in models:
        assert [m.joint(i).name for i in range(1, m.njnt)] == legs


def test_env_body_names_exist(models):
    for m in models:
        for b in ("pelvis", "torso_link", "left_payload", "right_payload"):
            m.body(b)


def test_qpos_order_equals_actuator_order(models):
    """env PD writes kp*(q* - qpos[7:]) straight into ctrl: the joint behind
    actuator i MUST be qpos[7 + i]."""
    for m in models:
        for i in range(m.nu):
            jid = m.actuator_trnid[i, 0]
            assert m.jnt_qposadr[jid] == 7 + i, m.actuator(i).name


def test_keyframe_is_default_pose(models):
    pose = np.asarray(load_config().actuators.default_pose)
    for m in models:
        np.testing.assert_allclose(m.keyframe("home").qpos[7:], pose, atol=1e-6)


def test_timestep_matches_control_config(models):
    dt = load_config().actuators.control.physics_dt
    for m in models:
        assert m.opt.timestep == pytest.approx(dt)


def test_ordering_identical(models):
    mjx_m, full_m = models
    assert ([mjx_m.joint(i).name for i in range(mjx_m.njnt)]
            == [full_m.joint(i).name for i in range(full_m.njnt)])
    assert ([mjx_m.actuator(i).name for i in range(mjx_m.nu)]
            == [full_m.actuator(i).name for i in range(full_m.nu)])


def test_payload_bodies(models):
    for m in models:
        for side in ("left", "right"):
            b = m.body(f"{side}_payload")
            assert m.body_mass[b.id] < 0.1  # nominal small mass


def test_home_keyframe_feet_on_floor(models):
    mjx_m, _ = models
    d = mujoco.MjData(mjx_m)
    mujoco.mj_resetDataKeyframe(mjx_m, d, 0)
    mujoco.mj_forward(mjx_m, d)
    for foot in ("left_foot", "right_foot"):
        gid = mjx_m.geom(foot).id
        sole = d.geom_xpos[gid][2] - mjx_m.geom_size[gid][2]
        # The keyframe starts slightly above the floor so episode-init is clean.
        # Sole must not be buried (> -5 mm) and not floating too high (< 30 mm).
        assert -5e-3 < sole < 3e-2


def test_reduced_contact_pairs(models):
    """Only the spec'd pairs can collide: foot-floor, foot-foot, hand-floor,
    hand-torso."""
    m, _ = models
    coll = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
    names = sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in coll)
    assert names == ["floor", "left_foot", "left_foot_self", "left_hand",
                     "right_foot", "right_foot_self", "right_hand",
                     "torso_capsule"]

    def can_collide(i, j):
        return bool((m.geom_contype[i] & m.geom_conaffinity[j])
                    or (m.geom_contype[j] & m.geom_conaffinity[i]))

    gid = {n: m.geom(n).id for n in names}
    assert can_collide(gid["left_foot"], gid["floor"])
    assert can_collide(gid["left_foot_self"], gid["right_foot_self"])  # foot-foot
    assert can_collide(gid["left_hand"], gid["floor"])
    assert can_collide(gid["left_hand"], gid["torso_capsule"])
    assert not can_collide(gid["left_foot"], gid["right_foot"])  # spheres own this
    assert not can_collide(gid["left_foot_self"], gid["floor"])
    assert not can_collide(gid["left_hand"], gid["right_hand"])
    assert not can_collide(gid["torso_capsule"], gid["floor"])
    assert not can_collide(gid["left_foot"], gid["torso_capsule"])


def test_com_centered_over_support(models):
    m, _ = models
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    com = d.subtree_com[m.body("pelvis").id]
    centroid = (d.geom_xpos[m.geom("left_foot").id]
                + d.geom_xpos[m.geom("right_foot").id]) / 2
    # Bent knees shift the CoM slightly off the sole-box centroid; 6 cm is ok.
    assert np.linalg.norm(com[:2] - centroid[:2]) < 0.06
