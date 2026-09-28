"""Drive the mapping route with Nav2 while slam_toolbox builds the map.

  ros2 run x1_bringup explore        (after: ros2 launch x1_bringup mapping.launch.py)
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from x1_bringup.places import EXPLORE_ROUTE


def pose(nav, x, y, yaw):
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x, p.pose.position.y = x, y
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


def main():
    rclpy.init()
    nav = BasicNavigator()
    # slam_toolbox is not a lifecycle node, and Humble's helper always waits
    # for the localiser's lifecycle state: wait on bt_navigator instead.
    nav.waitUntilNav2Active(localizer="bt_navigator")
    done = 0
    for i, (x, y, yaw) in enumerate(EXPLORE_ROUTE):
        nav.get_logger().info(f"waypoint {i + 1}/{len(EXPLORE_ROUTE)}: ({x:.1f}, {y:.1f})")
        nav.goToPose(pose(nav, x, y, yaw))
        while not nav.isTaskComplete():
            rclpy.spin_once(nav, timeout_sec=0.5)
        result = nav.getResult()
        if result == TaskResult.SUCCEEDED:
            done += 1
        else:
            nav.get_logger().warn(f"waypoint {i + 1} not reached ({result}); continuing")
    nav.get_logger().info(f"route finished: {done}/{len(EXPLORE_ROUTE)} waypoints reached")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
