#!/usr/bin/env python
"""Two-arm squeeze IK (damped least squares) with the pelvis fixed at 0.613 m.

Two contact modes per arm:
  pad     - the pad-face site on the box side face at mid-height, the pad
            normal pointing at the box centre;
  forearm - the forearm capsule axis parallel to the side face, one capsule
            radius outside it (line contact), its midpoint at mid-height;
  tip     - the pad's distal-medial edge (closed fingertips) on the face at
            mid-height, any pad angle (the hands converge in a V).
In both modes the contact must lie on the face (4 cm from the front/back
edges); a weak term pulls it towards the box CoM in x.
Soft constraints: the box is shrunk by the squeeze depth; the contact element
(pad or forearm) may touch it, every other arm proxy must stay 1 cm clear;
no proxy within 5 mm of the torso/pelvis collision hulls.
Joint limits are hard (0.02 rad margin).

  python manip/ik.py      # sweep -> manip/out/ik_sweep.json
"""

import json
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BOX_HALF, JOINTS, Scene  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LIM_MARGIN = 0.02
TORSO_CLEAR = 0.005
BOX_CLEAR = 0.01            # non-contact arm proxies vs the box
EDGE_MARGIN = 0.04
POS_TOL, ANG_TOL_DEG, PEN_TOL = 0.01, 15.0, 0.003
W_NORMAL = 0.05             # m per unit normal error
W_PEN = 3.0
W_XPREF = {"pad": 0.03, "forearm": 0.03, "tip": 0.3}   # pull of the contact towards the CoM (x)

SEEDS = [
    [-1.2, -0.1, 0.0, 1.0, 0.0, 0.0],
    [-0.5, -0.1, 0.0, 1.4, 0.0, 0.0],
    [-1.57, 0.0, 0.0, 0.5, 0.0, 0.0],
    [-0.9, -0.2, 0.5, 1.2, -0.8, 0.0],
    [-0.9, -0.2, -0.5, 1.2, 0.8, 0.0],
    [-1.4, -0.3, 0.0, 0.3, 1.2, 0.5],
    [-1.4, -0.3, 0.0, 0.3, -1.2, -0.5],
    [0.5, -0.3, 0.0, 1.6, 0.0, 0.0],
    [0.3, -0.5, 0.5, 1.8, -1.0, 0.5],
    [0.3, -0.5, -0.5, 1.8, 1.0, -0.5],
    [-0.3, -0.6, 0.0, 2.0, 1.5, 0.7],
    [-0.3, -0.6, 0.0, 2.0, -1.5, -0.7],
    [-1.0, -0.4, -1.2, 1.5, 0.0, 0.0],
    [-1.0, -0.4, 1.2, 1.5, 0.0, 0.0],
]


def targets(centre, depth=0.0):
    """Per-side contact target: face plane y, inward normal, mid-height z and
    the usable x range of the face."""
    c = np.asarray(centre, float)
    hw = BOX_HALF[1] - depth
    base = {"z": c[2], "xc": c[0], "xlo": c[0] - BOX_HALF[0] + EDGE_MARGIN,
            "xhi": c[0] + BOX_HALF[0] - EDGE_MARGIN, "depth": depth}
    return {"left": dict(base, y=c[1] + hw, n=np.array([0.0, -1.0, 0.0])),
            "right": dict(base, y=c[1] - hw, n=np.array([0.0, 1.0, 0.0]))}


class ArmIK:
    def __init__(self, scene: Scene, mode="pad"):
        self.s = scene
        self.mode = mode
        m = scene.m
        self.body_gids = [g for g in range(m.ngeom) if m.geom_contype[g] == 1
                          and m.geom_bodyid[g] in (scene.torso_bid, scene.pelvis_bid)]
        self.side_gids = {side: [g for g in scene.arm_gids
                                 if m.body(m.geom_bodyid[g]).name.startswith(side)]
                          for side in ("left", "right")}
        self.col = {"left": np.arange(12, 18), "right": np.arange(18, 24)}
        part = "elbow_yaw_cap" if mode == "forearm" else "pad"
        self.contact_gid = {side: m.geom(f"{side}_{part}").id for side in ("left", "right")}
        self.lo = scene.arm_jnt_lo + LIM_MARGIN
        self.hi = scene.arm_jnt_hi - LIM_MARGIN
        self._ft = np.zeros(6)

    def _dist(self, g1, g2):
        return mujoco.mj_geomDistance(self.s.m, self.s.d, g1, g2, 0.2, self._ft)

    def forearm_axis(self, side):
        m, d = self.s.m, self.s.d
        g = m.geom(f"{side}_elbow_yaw_cap").id
        c, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        h = m.geom_size[g, 1]
        return c - h * R[:, 2], c + h * R[:, 2], m.geom_size[g, 0]

    def contact_point(self, side):
        """World point that should touch the face (pad face centre or the
        forearm's medial line midpoint) and the contact normal/axis error."""
        s = self.s
        if self.mode in ("pad", "tip"):
            k = 0 if side == "left" else 1
            site = s.pad_site[k] if self.mode == "pad" else s.tip_site[k]
            return s.d.site_xpos[site].copy(), s.d.site_xmat[site].reshape(3, 3)[:, 2].copy()
        a, b, rad = self.forearm_axis(side)
        return 0.5 * (a + b), (b - a) / np.linalg.norm(b - a)

    def set(self, side, qa):
        self.s.d.qpos[self.s.qadr[self.col[side]]] = qa
        mujoco.mj_kinematics(self.s.m, self.s.d)

    def residual(self, side, qa, t):
        self.set(side, qa)
        if self.mode == "pad":
            p, z = self.contact_point(side)
            eq = np.concatenate([[p[1] - t["y"], p[2] - t["z"]], W_NORMAL * (z - t["n"])])
            x = p[0]
        elif self.mode == "tip":
            p, _ = self.contact_point(side)
            eq = np.array([p[1] - t["y"], p[2] - t["z"]])
            x = p[0]
        else:
            a, b, rad = self.forearm_axis(side)
            y_axis = t["y"] - t["n"][1] * rad       # axis one radius outside the face
            mid = 0.5 * (a + b)
            eq = np.array([a[1] - y_axis, b[1] - y_axis, mid[2] - t["z"]])
            x = mid[0]
        ineq = [max(0.0, t["xlo"] - x), max(0.0, x - t["xhi"])]
        m, bg = self.s.m, self.s.box_gid
        size0 = m.geom_size[bg].copy()
        contact_g = self.contact_gid[side]
        # the box is shrunk by the squeeze depth (the commanded pose sits d
        # inside; the real arm is stopped at the face by the contact element).
        # The contact element may touch it, every other proxy stays BOX_CLEAR out.
        m.geom_size[bg, 1] = size0[1] - t["depth"]
        for g in self.side_gids[side]:
            clear = 0.0 if g == contact_g else BOX_CLEAR
            ineq.append(W_PEN * max(0.0, clear - self._dist(g, bg)))
            for tg in self.body_gids:
                ineq.append(W_PEN * max(0.0, TORSO_CLEAR - self._dist(g, tg)))
        m.geom_size[bg] = size0
        return np.concatenate([eq, np.asarray(ineq), [W_XPREF[self.mode] * (x - t["xc"])]])

    def solve(self, side, t, seeds=None, iters=150):
        best = None
        for seed in (seeds if seeds is not None else SEEDS):
            qa = np.clip(np.asarray(seed, float), self.lo[:6], self.hi[:6])
            lam, eps = 0.05, 1e-5
            r = self.residual(side, qa, t)
            cost = r @ r
            for _ in range(iters):
                J = np.empty((r.size, 6))
                for j in range(6):
                    dq = qa.copy()
                    dq[j] += eps
                    J[:, j] = (self.residual(side, dq, t) - r) / eps
                step = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(r.size), -r)
                qn = np.clip(qa + np.clip(step, -0.2, 0.2), self.lo[:6], self.hi[:6])
                rn = self.residual(side, qn, t)
                cn = rn @ rn
                if cn < cost - 1e-12:
                    qa, r, cost = qn, rn, cn
                    lam = max(lam * 0.5, 1e-3)
                else:
                    lam = min(lam * 3.0, 10.0)
                    if lam >= 10.0:
                        break
            info = self.check(side, qa, t)
            if best is None or info["score"] < best[1]["score"]:
                best = (qa, info)
            if info["ok"]:
                break
        self.set(side, best[0])
        return best

    def check(self, side, qa, t):
        r = self.residual(side, qa, t)
        p, v = self.contact_point(side)
        if self.mode in ("pad", "tip"):
            pos = float(np.linalg.norm(r[:2]))
            ang = float(np.degrees(np.arccos(np.clip(v @ t["n"], -1, 1))))
            n_eq = 5 if self.mode == "pad" else 2
        else:
            pos = float(np.abs(r[:3]).max())
            ang = float(np.degrees(np.arcsin(min(1.0, abs(v @ t["n"])))))
            n_eq = 3
        ineq = r[n_eq:-1]
        edge = float(max(ineq[0], ineq[1]))
        pen = float(ineq[2:].max() / W_PEN) if ineq.size > 2 else 0.0
        ok = (pos < POS_TOL and pen < PEN_TOL and edge < 0.005
              and (ang < ANG_TOL_DEG or self.mode == "tip"))
        return {"ok": bool(ok), "pos_err": pos, "ang_err": ang, "pen": pen, "edge": edge,
                "x_contact": float(p[0]), "z_contact": float(p[2]),
                "score": pos + 0.002 * ang + pen + edge}


def box_body_clearance(scene, centre):
    """Signed clearance from the box to the legs, pelvis and torso, from the
    visual mesh vertices (the legs have no collision geometry). Negative =
    overlap depth."""
    s, m = scene, scene.m
    s.place_box(centre)
    mujoco.mj_kinematics(m, s.d)
    best, who = 1.0, ""
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or b == 0:
            continue
        name = m.body(b).name
        if any(k in name for k in ("shoulder", "shoudler", "elbow", "wrist")):
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = v @ s.d.geom_xmat[g].reshape(3, 3).T + s.d.geom_xpos[g] - np.asarray(centre)
        q = np.abs(w) - BOX_HALF
        sd = float((np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)).min())
        if sd < best:
            best, who = sd, name
    return best, who


def solve_squeeze(ik, centre, depth=0.0, seed=None):
    """Both arms. Returns (q_arm(12), ok, infos)."""
    t = targets(centre, depth)
    ik.s.place_box(centre)
    out, infos = np.zeros(12), {}
    for k, side in enumerate(("left", "right")):
        seeds = SEEDS if seed is None else [seed[6 * k:6 * k + 6]] + SEEDS
        if side == "right":
            seeds = [out[:6]] + seeds   # same sign convention: mirror the left arm
        qa, info = ik.solve(side, t[side], seeds=seeds)
        out[6 * k:6 * k + 6] = qa
        infos[side] = info
    return out, infos["left"]["ok"] and infos["right"]["ok"], infos


def sweep_row(args):
    hb, mode = args
    s = Scene(fixed=True)
    ik = ArmIK(s, mode)
    rows = []
    zc = hb + BOX_HALF[2]
    for xc in np.round(np.arange(0.36, 0.761, 0.02), 3):
        clear, who = box_body_clearance(s, [xc, 0.0, zc])
        row = {"mode": mode, "hb": hb, "xc": float(xc), "body_clear": round(clear, 4),
               "body_hit": who}
        if clear < 0.0:
            rows.append(dict(row, ok=False))
            continue
        q, ok, infos = solve_squeeze(ik, [xc, 0.0, zc])
        rows.append(dict(row, ok=bool(ok), q=np.round(q, 4).tolist(),
                         info={k: {kk: (round(v, 4) if isinstance(v, float) else v)
                                   for kk, v in i.items()} for k, i in infos.items()}))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    hbs = [round(h, 3) for h in np.arange(0.40, 1.201, 0.05)]
    jobs = [(hb, mode) for mode in ("pad", "forearm", "tip") for hb in hbs]
    with Pool(min(len(jobs), 20)) as p:
        res = p.map(sweep_row, jobs)
    rows = [r for rr in res for r in rr]
    with open(os.path.join(OUT, "ik_sweep.json"), "w") as f:
        json.dump({"joints": JOINTS[12:], "rows": rows}, f, indent=1)
    for mode in ("pad", "forearm", "tip"):
        print(f"== {mode}")
        for hb in hbs:
            rr = [r for r in rows if r["hb"] == hb and r["mode"] == mode]
            ok = [r for r in rr if r["ok"]]
            free = [r["xc"] for r in rr if r["body_clear"] >= 0]
            if ok:
                b = ok[0]
                print(f"box bottom {hb:.2f}: ok xc {ok[0]['xc']:.2f}..{ok[-1]['xc']:.2f} "
                      f"(n={len(ok)}), closest: contact x={b['info']['left']['x_contact']:.3f} "
                      f"free_from={min(free) if free else None}")
            else:
                best = min((r for r in rr if "info" in r),
                           key=lambda r: r["info"]["left"]["score"], default=None)
                msg = "" if best is None else (
                    f" best xc={best['xc']:.2f} " + str({k: v for k, v in best["info"]["left"].items()
                                                       if k in ("pos_err", "ang_err", "pen", "edge")}))
                print(f"box bottom {hb:.2f}: none (free_from={min(free) if free else None}){msg}")


if __name__ == "__main__":
    main()
