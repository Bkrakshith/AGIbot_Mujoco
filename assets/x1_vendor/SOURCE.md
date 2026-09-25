# AgiBot X1 vendor files (imported subset)

- Source: https://github.com/AgibotTech/agibot_x1_train @ e6651b9
  (resources/robots/x1/). License: BSD-3-Clause (header of humanoid/envs/x1/x1_dh_stand_config.py).
- `urdf/x1.urdf`: reference for link masses, joint axes and efforts. It is NOT
  loaded by the pipeline, and its 24 linkage/detail meshes (toe/waist/wrist
  motor links) were deliberately not imported, so it will not render fully.
- `mjcf/xyber_x1_serial.xml`: the vendor MuJoCo model (the one their sim2sim
  uses). `scripts/build_assets.py` generates assets/x1_*.xml from it.
- `meshes/`: only the 30 STL files that the MJCF references.
- Training parameters taken from humanoid/envs/x1/x1_dh_stand_config.py are
  cited inline in configs/*.yaml ("X1 cfg: ...").
