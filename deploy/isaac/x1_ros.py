"""ROS 2 interface for the X1 in Isaac Sim: sensors, state and commands.

Used by x1_office_teleop.py when started with --ros. Isaac Sim talks to ROS
through its own bundled Humble libraries, so start it from a terminal set up
with `source ros2_ws/scripts/ros2_env.sh --isaac` (see README).

Published
  /clock                    rosgraph_msgs/Clock      simulation time
  /mid360/points            sensor_msgs/PointCloud2  head lidar, frame mid360
  /d435/color/image_raw     sensor_msgs/Image        chest camera (RGB)
  /d435/depth/image_raw     sensor_msgs/Image        chest camera (depth, m)
  /d435/color/camera_info   sensor_msgs/CameraInfo
  /odom                     nav_msgs/Odometry        odom -> base_link (simulator ground truth)
  /tf, /tf_static           odom -> base_link, base_link -> sensor frames
  /joint_states             sensor_msgs/JointState   24 joints
Subscribed
  /cmd_vel                  geometry_msgs/Twist      walking command (vx, vy, yaw rate)
  /x1/arm_pose              std_msgs/String          arm pose name (down, forward, reach,
                                                     sideways, overhead, back, carry)
  /x1/arm_joints            std_msgs/Float64MultiArray  explicit 12 arm targets (rad)
  /x1/gripper               std_msgs/Bool            true = close both grippers
With --task, also published: /x1/grasped (std_msgs/Bool) and the tf frame "box".

base_link is the pelvis. The waist joints are fixed, so the sensors are rigid
with the pelvis and their frames are static. Mount poses come from
deploy/isaac/sensors/x1_sensors.yaml (D435 from the vendor CAD; the Mid-360
mount is the planned head bracket).

Sensor models: Isaac Sim ships no Livox model, so the head lidar is an Ouster
OS0 (128 channels, 360 x 90 deg, 10 Hz) at the Mid-360 mount, upside down.
Its field of view covers the Mid-360's (360 x 59 deg). Isaac also has no D435,
so the chest camera is a pinhole camera with the D435 depth field of view
(87 x 57 deg at 848 x 480).
"""

import math
import os

import numpy as np

# mounts in the pelvis frame (x forward, y left, z up)
MID360_XYZ = (0.037, 0.0, 0.666)          # upside down on the head
D435_XYZ = (0.0852, 0.0177, 0.4185)       # depth origin, looking straight ahead
D435_RES = (848, 480)
D435_HFOV_DEG = 87.0
OS0_VARIANT = "OS0_REV7_128ch10hz512res"   # lightest 10 Hz profile
CMD_TIMEOUT_S = 0.5                       # /cmd_vel older than this is ignored (sim time)


def _quat_wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


# camera prim: looks along -Z, +Y up. Map -Z -> +x (forward), +Y -> +z (up).
R_CAM_USD = np.array([[0.0, 0.0, -1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
# ROS optical frame: z forward, x right, y down
R_CAM_OPTICAL = np.array([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]])
R_LIDAR = np.diag([1.0, -1.0, -1.0])      # roll pi: upside down


def create_sensors(stage, pelvis_path):
    """Author the lidar and camera prims under the pelvis. Call before Play."""
    from isaacsim.sensors.experimental.rtx import Lidar
    from pxr import Gf, UsdGeom

    lidar = Lidar.create(path=f"{pelvis_path}/mid360", config="OS0", variant=OS0_VARIANT,
                         tick_rate=10.0, translations=[list(MID360_XYZ)],
                         orientations=[_quat_wxyz(R_LIDAR)])
    cam = UsdGeom.Camera.Define(stage, f"{pelvis_path}/d435")
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*D435_XYZ))
    w, x, y, z = _quat_wxyz(R_CAM_USD)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    h_ap = 20.0                                             # mm
    focal = h_ap / (2.0 * math.tan(math.radians(D435_HFOV_DEG) / 2.0))
    cam.GetHorizontalApertureAttr().Set(h_ap)
    cam.GetVerticalApertureAttr().Set(h_ap * D435_RES[1] / D435_RES[0])
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 20.0))
    return lidar, cam.GetPath().pathString


def start_sensor_publishers(lidar, cam_path):
    """Attach the ROS writers and build the clock + camera graph. Call after Play."""
    import omni.graph.core as og
    import usdrt.Sdf
    from isaacsim.sensors.experimental.rtx import LidarSensor

    sensor = LidarSensor(lidar, annotators=[])
    sensor.attach_writer("RtxLidarROS2PublishPointCloud", topicName="/mid360/points", frameId="mid360")
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/World/ROS", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("simTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("render", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("rgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "clock.inputs:execIn"),
                ("simTime.outputs:simulationTime", "clock.inputs:timeStamp"),
                ("tick.outputs:tick", "render.inputs:execIn"),
                ("render.outputs:execOut", "rgb.inputs:execIn"),
                ("render.outputs:execOut", "depth.inputs:execIn"),
                ("render.outputs:execOut", "info.inputs:execIn"),
                ("render.outputs:renderProductPath", "rgb.inputs:renderProductPath"),
                ("render.outputs:renderProductPath", "depth.inputs:renderProductPath"),
                ("render.outputs:renderProductPath", "info.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("render.inputs:cameraPrim", [usdrt.Sdf.Path(cam_path)]),
                ("render.inputs:width", D435_RES[0]),
                ("render.inputs:height", D435_RES[1]),
                ("rgb.inputs:type", "rgb"),
                ("rgb.inputs:topicName", "/d435/color/image_raw"),
                ("rgb.inputs:frameId", "d435_optical"),
                ("depth.inputs:type", "depth"),
                ("depth.inputs:topicName", "/d435/depth/image_raw"),
                ("depth.inputs:frameId", "d435_optical"),
                ("info.inputs:topicName", "/d435/color/camera_info"),
                ("info.inputs:frameId", "d435_optical"),
            ],
        },
    )
    return sensor


class RosBridge:
    """rclpy node inside Isaac Sim: robot state out, commands in."""

    def __init__(self, joint_names, arm_pose_names):
        import rclpy
        from geometry_msgs.msg import TransformStamped, Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Float64MultiArray, String
        from tf2_msgs.msg import TFMessage
        from rclpy.qos import DurabilityPolicy, QoSProfile

        if not rclpy.ok():
            rclpy.init()
        self.rclpy, self.TS, self.Odometry, self.JointState = rclpy, TransformStamped, Odometry, JointState
        self.TFMessage = TFMessage
        self.node = rclpy.create_node("x1_isaac")
        self.joint_names = list(joint_names)
        self.arm_pose_names = set(arm_pose_names)
        self.pub_odom = self.node.create_publisher(Odometry, "/odom", 10)
        self.pub_tf = self.node.create_publisher(TFMessage, "/tf", 10)
        static_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_tf_static = self.node.create_publisher(TFMessage, "/tf_static", static_qos)
        self.pub_js = self.node.create_publisher(JointState, "/joint_states", 10)
        self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.node.create_subscription(String, "/x1/arm_pose", self._on_arm_pose, 10)
        self.node.create_subscription(Float64MultiArray, "/x1/arm_joints", self._on_arm_joints, 10)
        self.node.create_subscription(Bool, "/x1/gripper", self._on_gripper, 10)
        self.pub_grasped = self.node.create_publisher(Bool, "/x1/grasped", 10)
        self.Bool = Bool
        self.gripper_request = None
        self.cmd, self.cmd_time = np.zeros(3), -1e9
        self.arm_request = None
        self.now = 0.0
        self.static_sent = False

    # ------------------------------------------------------------ commands
    def _on_cmd_vel(self, msg):
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.cmd_time = self.now

    def _on_arm_pose(self, msg):
        if msg.data in self.arm_pose_names:
            self.arm_request = msg.data

    def _on_arm_joints(self, msg):
        if len(msg.data) == 12:
            self.arm_request = np.array(msg.data, float)

    def _on_gripper(self, msg):
        self.gripper_request = bool(msg.data)

    def active_cmd(self):
        """The latest /cmd_vel, or None if nothing arrived recently."""
        return self.cmd if self.now - self.cmd_time < CMD_TIMEOUT_S else None

    # ------------------------------------------------------------- publish
    def _stamp(self, t):
        from builtin_interfaces.msg import Time
        sec = int(t)
        return Time(sec=sec, nanosec=int((t - sec) * 1e9))

    def _tf(self, stamp, parent, child, xyz, R):
        from scipy.spatial.transform import Rotation
        m = self.TS()
        m.header.stamp, m.header.frame_id, m.child_frame_id = stamp, parent, child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, xyz)
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        m.transform.rotation.x, m.transform.rotation.y = float(x), float(y)
        m.transform.rotation.z, m.transform.rotation.w = float(z), float(w)
        return m

    def publish(self, sim_time, pos, R, lin_world, ang_world, q, qd, box=None, grasped=None, tcps=None):
        self.now = sim_time
        stamp = self._stamp(sim_time)
        if not self.static_sent:
            self.pub_tf_static.publish(self.TFMessage(transforms=[
                self._tf(stamp, "base_link", "mid360", MID360_XYZ, R_LIDAR),
                self._tf(stamp, "base_link", "d435_optical", D435_XYZ, R_CAM_OPTICAL),
            ]))
            self.static_sent = True
        tf = self._tf(stamp, "odom", "base_link", pos, R)
        tfs = [tf]
        if box is not None:
            tfs.append(self._tf(stamp, "odom", "box", box[0], box[1]))
        for name, p in zip(("left_tcp", "right_tcp"), tcps or ()):
            tfs.append(self._tf(stamp, "odom", name, p, np.eye(3)))
        self.pub_tf.publish(self.TFMessage(transforms=tfs))
        if grasped is not None:
            self.pub_grasped.publish(self.Bool(data=bool(grasped)))
        od = self.Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, "odom", "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = map(float, pos)
        od.pose.pose.orientation = tf.transform.rotation
        v, w = R.T @ lin_world, R.T @ ang_world          # twist in base_link
        od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = map(float, v)
        od.twist.twist.angular.x, od.twist.twist.angular.y, od.twist.twist.angular.z = map(float, w)
        self.pub_odom.publish(od)
        js = self.JointState()
        js.header.stamp = stamp
        js.name = self.joint_names
        js.position, js.velocity = [float(v) for v in q], [float(v) for v in qd]
        self.pub_js.publish(js)

    def spin_once(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self):
        self.node.destroy_node()
