from setuptools import find_packages, setup

package_name = "semantic_nav"

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
    description="Natural-language semantic navigation for ROS 2 / Nav2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_nav_node = semantic_nav.semantic_nav_node:main",
            "safety_node = semantic_nav.safety_node:main",
            "landmark_markers_node = semantic_nav.landmark_markers_node:main",
            "semantic_nav_cli = semantic_nav.cli:main",
        ],
    },
)
