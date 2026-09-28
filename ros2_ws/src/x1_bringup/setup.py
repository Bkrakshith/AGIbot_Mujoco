import os
from glob import glob

from setuptools import setup

package_name = "x1_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "maps"), glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="AgiBot X1 office SLAM, navigation and task nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "explore = x1_bringup.explore:main",
            "save_map = x1_bringup.save_map:main",
            "fetch_box = x1_bringup.fetch_box:main",
        ],
    },
)
