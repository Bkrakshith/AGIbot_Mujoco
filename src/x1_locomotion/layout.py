"""Robot DoF layout, read once from configs/actuators.yaml (`layout:`).

Every size and index that depends on the robot comes from here so the env,
the CPU evaluator, the export path and the tests cannot disagree. AgiBot X1:
12 leg fully-learned DoF, no arm or waist joints (fixed in the vendor model).
"""

import os

import yaml

_ACT = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "actuators.yaml")
with open(_ACT) as _f:
    _L = yaml.safe_load(_f)["layout"]

N_LT = int(_L["n_legs_waist"])            # legs + waist, full authority
N_ARM = int(_L["n_arm"])                  # arm residual joints
NU = N_LT + N_ARM                         # actuated joints = action size
N_LEG = int(_L["leg_joints"])             # both legs
LEG_QPOS = slice(7, 7 + N_LEG)            # leg joints in qpos
LT_QPOS = slice(7, 7 + N_LT)
ARM_QPOS = slice(7 + N_LT, 7 + NU)
KNEE_QPOS = tuple(7 + i for i in _L["knee_idx"])
HIP_ROLL_IDX = tuple(int(i) for i in _L["hip_roll_idx"])
