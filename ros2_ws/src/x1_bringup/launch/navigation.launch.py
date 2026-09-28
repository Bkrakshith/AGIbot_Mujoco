"""Navigate in the saved office map: head-lidar scan + slam_toolbox
localisation + Nav2.

  ros2 launch x1_bringup navigation.launch.py [map:=<path without extension>] [rviz:=true]
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
    map_arg = LaunchConfiguration("map")
    return LaunchDescription([
        DeclareLaunchArgument("map", default_value=os.path.join(share, "maps", "office")),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node",
             name="pointcloud_to_laserscan", parameters=[cfg("pointcloud_to_laserscan.yaml")],
             remappings=[("cloud_in", "/mid360/points"), ("scan", "/scan")]),
        Node(package="slam_toolbox", executable="localization_slam_toolbox_node", name="slam_toolbox",
             parameters=[cfg("slam_localization.yaml"), {"map_file_name": map_arg}], output="screen"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2, "launch", "navigation_launch.py")),
            launch_arguments={"use_sim_time": "true", "params_file": cfg("nav2_x1.yaml"),
                              "autostart": "true"}.items()),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(share, "config", "x1.rviz")],
             parameters=[{"use_sim_time": True}], condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
