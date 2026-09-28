"""Named places in the Isaac Sim office, in map coordinates (m, rad).

The map frame lines up with the simulator's world frame (odometry is the
simulator's ground truth and slam_toolbox starts the map at the first odom
pose), so these come straight from deploy/isaac/office_map.yaml.
"""

SPAWN = (-17.5, 30.5, 0.0)

# Mapping route through the hall, the east office and the east meeting room,
# through the doors at (-5.0, 28.5) and (-3.0, 35.5), then back to the hall.
# Every waypoint keeps >= 1 m from walls (outside the costmap inflation).
EXPLORE_ROUTE = [
    (-14.0, 30.5, 0.0),
    (-11.0, 31.0, 0.0),
    (-8.0, 31.0, 0.0),
    (-6.5, 28.8, -0.3),
    (-3.6, 28.8, 0.0),
    (-3.0, 32.0, 1.5708),
    (-3.0, 37.2, 1.5708),
    (0.0, 40.0, 0.0),
    (-3.0, 33.5, -1.5708),
    (-6.5, 28.8, 3.1416),
    (-11.0, 29.5, 3.1416),
    (-16.5, 30.5, 3.1416),
]

RECEPTION_NORTH_STAND = (-11.90, 28.70, -1.5708)
MEETING_EAST_STAND = (2.95, 36.85, 1.5708)

# Box task (deploy/isaac/x1_task.py). The pickup table is a 0.75 m table next
# to the north reception; the box is placed on the south end of the east
# meeting-room table. The robot stands so the box centre is HOLD_X in front
# of the pelvis: that is where the trained "carry" arm pose puts the grippers.
HOLD_X = 0.49
TABLE_TOP = 0.75
# box centre on the south end of the east meeting-room table (top 0.75 m,
# end edge at y 37.33), 0.17 m in from the edge; the robot faces north (+y).
# The room wall is ~1.8 m behind the stance, so the place walk-in is short.
PLACE_BOX = (2.95, 37.50, 1.5708)
PICK_APPROACH_BACK = 2.0                     # m behind the stance where Nav2 stops, on the approach line
PLACE_APPROACH_BACK = 0.5
# How far the robot keeps walking after the walk-in command stops (measured in
# Isaac with the current gait): a few cm; the previous gait carried on ~0.12 m.
PICK_STOP_DRIFT = 0.03
PLACE_STOP_DRIFT = 0.02

# Arm poses (one arm; both arms use the same values) for a box resting on a
# surface at `top_above_pelvis`: measured in manip/handle_box.yaml, all within
# the walking policy's trained arm-pose envelope (the "carry" pose family).
PLACE_POSES = [  # (top_above_pelvis, [shoulder pitch, roll, yaw, elbow, elbow yaw, wrist])
    (0.0743, [-0.20, -0.1, 0.0, 1.4, 0.0, -0.30]),
    (0.1110, [-0.35, -0.1, 0.0, 1.4, 0.0, -0.15]),
    (0.1528, [-0.50, -0.1, 0.0, 1.4, 0.0, 0.00]),
    (0.1991, [-0.65, -0.1, 0.0, 1.4, 0.0, 0.15]),
    (0.2493, [-0.80, -0.1, 0.0, 1.4, 0.0, 0.30]),
]
LIFT = 0.06                                   # m, box lifted this far above the surface to carry

# Gripper-centre Jacobians (pelvis frame, rows x/y/z, columns = the six arm
# joints of that arm) at the pick pose for a 0.75 m table, from the MuJoCo
# model. Used to nudge the arms onto the handles after the walk-in.
J_LEFT = [[-0.0407, 0.0069, -0.0100, -0.1034, -0.0019, 0.0322],
          [0.0000, -0.2170, 0.3502, -0.0351, 0.0042, 0.0099],
          [-0.4115, -0.0142, 0.0342, 0.3392, 0.0062, -0.0942]]
J_RIGHT = [[-0.0409, 0.0069, -0.0100, -0.1032, -0.0019, 0.0322],
           [0.0000, 0.2172, -0.3501, 0.0351, -0.0040, -0.0099],
           [-0.4115, -0.0142, 0.0342, 0.3392, 0.0061, -0.0942]]
HANDLE_Y, HANDLE_Z = 0.2218, 0.045       # handle centres in the box frame (manip/handle_box.yaml)
ARM_NUDGE_MAX = 0.5                      # rad, largest change from the base pick pose per joint
