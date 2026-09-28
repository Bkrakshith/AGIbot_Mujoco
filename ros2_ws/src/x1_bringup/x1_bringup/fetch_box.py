"""Fetch the handle box from the pickup table and put it on the meeting table.

  ros2 run x1_bringup fetch_box
Needs Isaac Sim running x1_office_teleop.py --ros --task, and
ros2 launch x1_bringup navigation.launch.py (or mapping.launch.py).

Steps: Nav2 to the table -> line up on the box (tf "box") -> arms to the pick
pose -> close grippers -> lift -> Nav2 to the meeting table -> line up on the
place spot -> lower -> open -> step back -> arms down.
"""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_msgs.msg import Bool, Float64MultiArray, String
from tf2_ros import Buffer, TransformListener

from x1_bringup import places as P




def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rot(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def arm_pose_for(top_above_pelvis):
    """Interpolate the measured place poses for a surface height."""
    hs = [h for h, _ in P.PLACE_POSES]
    qs = np.array([q for _, q in P.PLACE_POSES])
    h = float(np.clip(top_above_pelvis, hs[0], hs[-1]))
    one = np.array([np.interp(h, hs, qs[:, j]) for j in range(6)])
    return np.concatenate([one, one])


class FetchBox(BasicNavigator):
    def __init__(self):
        super().__init__("fetch_box")
        # all waits and timeouts are in simulation time (the sim runs slower than real time)
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.arm = self.create_publisher(Float64MultiArray, "/x1/arm_joints", 10)
        self.arm_named = self.create_publisher(String, "/x1/arm_pose", 10)
        self.grip = self.create_publisher(Bool, "/x1/gripper", 10)
        self.grasped = False
        self.create_subscription(Bool, "/x1/grasped", lambda m: setattr(self, "grasped", m.data), 10)

    # ---------------------------------------------------------------- helpers
    def wait(self, sim_s):
        """Wait `sim_s` seconds of SIMULATION time (the sim runs slower than real time)."""
        end = self.get_clock().now() + Duration(seconds=sim_s)
        while self.get_clock().now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def lookup(self, parent, child):
        while True:
            try:
                t = self.tf.lookup_transform(parent, child, Time())
                return t.transform
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)

    def pelvis_height(self):
        return self.lookup("odom", "base_link").translation.z

    def goto(self, x, y, yaw, label):
        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        for attempt in range(3):
            self.info(f"{label}: navigating to ({x:.2f}, {y:.2f})")
            self.goToPose(p)
            while not self.isTaskComplete():
                rclpy.spin_once(self, timeout_sec=0.2)
            if self.getResult() == TaskResult.SUCCEEDED:
                return True
            self.warn(f"{label}: Nav2 attempt {attempt + 1} failed")
        return False

    def _errors(self, target_in_frame, frame):
        """(along, cross, yaw) error of the pelvis from its stance for the target,
        measured along and across the approach line (the target's heading):
        along > 0 = stance still ahead, cross > 0 = line is to the robot's left."""
        tx, ty, tyaw = target_in_frame
        base = self.lookup(frame, "base_link")
        bx, by, byaw = base.translation.x, base.translation.y, yaw_of(base.rotation)
        gx, gy = tx - P.HOLD_X * math.cos(tyaw), ty - P.HOLD_X * math.sin(tyaw)
        dx, dy = gx - bx, gy - by
        return (math.cos(tyaw) * dx + math.sin(tyaw) * dy,
                -math.sin(tyaw) * dx + math.cos(tyaw) * dy,
                wrap(tyaw - byaw))

    def _move(self, vx=0.0, vy=0.0, wz=0.0, duration=0.0):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(wz)
        end = self.get_clock().now() + Duration(seconds=duration)
        while self.get_clock().now() < end:
            self.cmd.publish(m)
            self.wait(0.05)
        self.stop()
        self.wait(1.0)                                   # let the gait settle

    def face(self, target_in_frame, frame, tol=0.05, timeout_s=15.0):
        """Turn in place until the robot faces along the approach line."""
        end = self.get_clock().now() + Duration(seconds=timeout_s)
        m = Twist()
        while self.get_clock().now() < end:
            _, _, eyaw = self._errors(target_in_frame, frame)
            if abs(eyaw) < tol:
                break
            m.angular.z = math.copysign(0.3 if abs(eyaw) > 0.2 else 0.2, eyaw)
            self.cmd.publish(m)
            self.wait(0.05)
        self.stop()
        self.wait(1.0)

    def align(self, target_in_frame, frame, label, stop_drift, tol_xy=0.20, tol_yaw=0.16):
        """Walk onto the stance that puts the target HOLD_X straight ahead of the
        pelvis, facing the same way. Call from ~1 m back on the approach line.

        The walking policy tracks steady forward walking and turning on the spot
        well, but not short sideways or forward nudges (it shifts its weight
        first). So: turn to face along the line, then walk forward at 0.2 m/s
        with Stanley steering (heading error + atan of the cross-track error) and
        stop STOP_DRIFT short, since the gait carries on after the command ends."""
        self.face(target_in_frame, frame)
        ex, ey, eyaw = self._errors(target_in_frame, frame)
        self.info(f"{label}: approach from dx {ex * 100:.1f} cm, dy {ey * 100:.1f} cm, "
                  f"dyaw {math.degrees(eyaw):.1f} deg")
        m = Twist()
        next_log = self.get_clock().now()
        while True:
            ex, ey, eyaw = self._errors(target_in_frame, frame)
            if self.get_clock().now() >= next_log:
                self.info(f"{label}:   along {ex * 100:.0f} cm, cross {ey * 100:.1f} cm, "
                          f"heading {math.degrees(eyaw):.1f} deg, turn {m.angular.z:+.2f}")
                next_log = self.get_clock().now() + Duration(seconds=1.0)
            if ex < stop_drift:
                break
            # Walking already (gait active), so a sideways component in the
            # command is tracked; it corrects the cross-track error without
            # turning, while the turn command holds the heading. Slow down over
            # the last 0.4 m so the robot stops shorter and more repeatably.
            far = float(np.clip((ex - stop_drift) / 0.4, 0.0, 1.0))
            m.linear.x = 0.12 + 0.08 * far
            m.linear.y = float(np.clip(2.5 * ey, -0.2, 0.2))
            m.angular.z = float(np.clip(2.0 * eyaw, -0.4, 0.4))
            self.cmd.publish(m)
            self.wait(0.05)
        self.stop()
        self.wait(1.5)
        ex, ey, eyaw = self._errors(target_in_frame, frame)
        msg = f"dx {ex * 100:.1f} cm, dy {ey * 100:.1f} cm, dyaw {math.degrees(eyaw):.1f} deg"
        ok = abs(ex) < tol_xy and abs(ey) < tol_xy and abs(eyaw) < tol_yaw
        (self.info if ok else self.warn)(f"{label}: {'aligned' if ok else 'missed'} ({msg})")
        return ok

    def approach(self, target_xyyaw_map, align_target, align_frame, label, approach_back, stop_drift, tries=3):
        """Nav2 to APPROACH_BACK behind the stance, then align; retry via Nav2."""
        tx, ty, tyaw = target_xyyaw_map
        back = P.HOLD_X + approach_back
        for _ in range(tries):
            if not self.goto(tx - back * math.cos(tyaw), ty - back * math.sin(tyaw), tyaw, label):
                continue
            if self.align(align_target(), align_frame, label, stop_drift):
                return True
        return False

    def reach_handles(self, q_base, rounds=4, tol=0.025):
        """Nudge each arm so its gripper centre sits on its handle (closed loop on
        the gripper-centre frames Isaac publishes; covers arm sag and the few
        centimetres the walk-in leaves)."""
        q = np.array(q_base, float)
        J = [np.array(P.J_LEFT), np.array(P.J_RIGHT)]
        for r in range(rounds):
            box = self.lookup("base_link", "box")
            Rb = _rot(box.rotation)
            pb = np.array([box.translation.x, box.translation.y, box.translation.z])
            err = []
            for i, (side, sy) in enumerate((("left", 1.0), ("right", -1.0))):
                h = pb + Rb @ np.array([0.0, sy * P.HANDLE_Y, P.HANDLE_Z])
                t = self.lookup("base_link", f"{side}_tcp").translation
                e = h - np.array([t.x, t.y, t.z])
                err.append(e)
                Ji = J[i]
                dq = Ji.T @ np.linalg.solve(Ji @ Ji.T + 0.02 * np.eye(3), e)   # damped least squares
                q[6 * i:6 * i + 6] = np.clip(q[6 * i:6 * i + 6] + 0.8 * dq,
                                             q_base[6 * i:6 * i + 6] - P.ARM_NUDGE_MAX,
                                             q_base[6 * i:6 * i + 6] + P.ARM_NUDGE_MAX)
            n = [float(np.linalg.norm(e)) for e in err]
            self.info(f"arms: round {r + 1}, gripper-to-handle {n[0] * 100:.1f} / {n[1] * 100:.1f} cm")
            if max(n) < tol:
                break
            self.arms(q, settle_s=1.5)
        return q

    def stop(self):
        for _ in range(5):
            self.cmd.publish(Twist())
            self.wait(0.05)

    def arms(self, q, settle_s=2.0):
        self.arm.publish(Float64MultiArray(data=[float(v) for v in q]))
        self.wait(settle_s)

    def gripper(self, close):
        self.grip.publish(Bool(data=close))
        self.wait(0.8)
        return self.grasped

    # ------------------------------------------------------------------- task
    def run(self):
        self.waitUntilNav2Active(localizer="bt_navigator")   # slam_toolbox is not a lifecycle node
        self.arm_named.publish(String(data="down"))
        # approach goal from the box's own pose in the map frame (so any offset
        # between the SLAM map and odometry cancels out); align on its odom pose
        bm = self.lookup("map", "box")

        def box_odom():
            b = self.lookup("odom", "box")
            return (b.translation.x, b.translation.y, yaw_of(b.rotation))

        if not self.approach((bm.translation.x, bm.translation.y, yaw_of(bm.rotation)), box_odom, "odom", "pick",
                             P.PICK_APPROACH_BACK, P.PICK_STOP_DRIFT):
            return False
        box = self.lookup("odom", "box")
        top_rel = (box.translation.z - 0.07) - self.pelvis_height()
        pick = arm_pose_for(top_rel)
        self.arms(pick)
        pick = self.reach_handles(pick)
        if not self.gripper(True):
            self.error("grasp failed: grippers too far from the handles")
            return False
        self.info("box held; lifting")
        self.arms(arm_pose_for(top_rel + P.LIFT))
        if not self.approach(P.PLACE_BOX, lambda: P.PLACE_BOX, "map", "place", P.PLACE_APPROACH_BACK,
                             P.PLACE_STOP_DRIFT):
            return False
        self.arms(arm_pose_for(P.TABLE_TOP - self.pelvis_height() + 0.005), settle_s=2.5)
        self.gripper(False)
        self.info("box placed; stepping back")
        back = Twist()
        back.linear.x = -0.15
        end = self.get_clock().now() + Duration(seconds=2.0)
        while self.get_clock().now() < end:
            self.cmd.publish(back)
            self.wait(0.05)
        self.stop()
        self.arm_named.publish(String(data="down"))
        self.info("done")
        return True


def main():
    rclpy.init()
    node = FetchBox()
    ok = node.run()
    node.info(f"fetch_box finished: {'success' if ok else 'FAILED'}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
