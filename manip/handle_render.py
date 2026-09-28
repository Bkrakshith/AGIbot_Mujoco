#!/usr/bin/env python
"""Video of the handle-box task: pick from the table, walk 0.2 m/s, turn,
place on the "midhigh" table, step back. Same episode as handle_task.py,
with frames captured at 30 fps.

  MUJOCO_GL=egl python manip/handle_render.py   # -> outputs/videos/manip/handle_box_carry.mp4
"""

import os
import sys

import mediapy as media
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import handle_task as H  # noqa: E402
from common import VIDEO_DIR, renderer, snap  # noqa: E402

FPS = 30
PLACE, SEED, VX = "midhigh", 3001, 0.2


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    frames, stills = [], {}
    state = {}
    orig_init = H.Task.__init__

    def init(self, seed, spec):
        orig_init(self, seed, spec)
        ren, opt, cam = renderer(self.s, 960, 540)
        cam.type = 1                      # tracking camera on the pelvis
        cam.trackbodyid = self.s.pelvis_bid
        cam.distance, cam.elevation, cam.azimuth = 2.3, -14.0, 150.0
        state.update(ren=ren, opt=opt, cam=cam, k=0, task=self)
        step = self.r.step

        def rec(arm, cmd=(0.0, 0.0, 0.0)):
            ok = step(arm, cmd)
            state["k"] += 1
            if state["k"] % int(round(1.0 / (FPS * 0.01))) == 0:
                frames.append(snap(self.s, ren, opt, cam))
                stills[self.stage] = frames[-1]
            return ok
        self.r.step = rec
    H.Task.__init__ = init
    res = H.episode((PLACE, SEED, VX))
    try:
        import imageio_ffmpeg
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    media.write_video(os.path.join(VIDEO_DIR, "handle_box_carry.mp4"), frames, fps=FPS)
    keys = [k for k in ("grasp", "walk", "turn", "step_back") if k in stills]
    media.write_image(os.path.join(VIDEO_DIR, "handle_box_stills.png"),
                      np.concatenate([stills[k] for k in keys], 1))
    print(f"{len(frames)} frames, stages {res['stages']}")


if __name__ == "__main__":
    main()
