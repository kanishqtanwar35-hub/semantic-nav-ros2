from setuptools import find_packages, setup

package_name = "robot_core"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kanishq Tanwar",
    maintainer_email="kanishqtanwar35@gmail.com",
    description="ROS-free robotics algorithms: mapping, planning, geometry, safety.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
