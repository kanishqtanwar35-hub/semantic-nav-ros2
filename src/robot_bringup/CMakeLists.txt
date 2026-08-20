cmake_minimum_required(VERSION 3.8)
project(robot_bringup)

find_package(ament_cmake REQUIRED)

# An ament_cmake package that installs data only. It exists rather than folding
# these files into semantic_nav because the description, the world and the
# launch files are what changes when the robot changes, and keeping them
# separate means swapping simulation for hardware touches one package.
install(
  DIRECTORY launch urdf config worlds maps
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  # Checks the description parses, the launch files are importable and the
  # generated maps match the geometry the unit tests plan against.
  ament_add_pytest_test(bringup_tests test
    TIMEOUT 180
  )
endif()

ament_package()
