"""Save the slam_toolbox map twice: as a pose graph (for slam_toolbox
localisation) and as an occupancy image (for viewing / map_server).

  ros2 run x1_bringup save_map [path without extension]
Default path: ros2_ws/src/x1_bringup/maps/office (rebuild the workspace after
saving so the launch files find it in the install space).
"""
import os
import subprocess
import sys

import rclpy
from slam_toolbox.srv import SerializePoseGraph


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "src", "x1_bringup", "maps", "office"))
    target = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.environ.get("X1_MAP", default)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("x1_save_map")
    cli = node.create_client(SerializePoseGraph, "/slam_toolbox/serialize_map")
    if not cli.wait_for_service(timeout_sec=10.0):
        node.get_logger().error("slam_toolbox is not running")
        sys.exit(1)
    fut = cli.call_async(SerializePoseGraph.Request(filename=target))
    rclpy.spin_until_future_complete(node, fut, timeout_sec=60.0)
    node.get_logger().info(f"pose graph -> {target}.posegraph / .data")
    subprocess.run(["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", target,
                    "--ros-args", "-p", "use_sim_time:=true"], check=False)
    node.get_logger().info(f"occupancy map -> {target}.pgm / .yaml")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
