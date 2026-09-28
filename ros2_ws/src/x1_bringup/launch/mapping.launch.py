"""Map the office: head-lidar scan + slam_toolbox + Nav2 (drive while mapping).

  ros2 launch x1_bringup mapping.launch.py [rviz:=true]
Save the map afterwards with: ros2 run x1_bringup save_map
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("x1_bringup")
    nav2 = get_package_share_directory("nav2_bringup")
    cfg = lambda f: os.path.join(share, "config", f)
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node",
             name="pointcloud_to_laserscan", parameters=[cfg("pointcloud_to_laserscan.yaml")],
             remappings=[("cloud_in", "/mid360/points"), ("scan", "/scan")]),
        Node(package="slam_toolbox", executable="async_slam_toolbox_node", name="slam_toolbox",
             parameters=[cfg("slam_mapping.yaml")], output="screen"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2, "launch", "navigation_launch.py")),
            launch_arguments={"use_sim_time": "true", "params_file": cfg("nav2_x1.yaml"),
                              "autostart": "true"}.items()),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(share, "config", "x1.rviz")],
             parameters=[{"use_sim_time": True}], condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
