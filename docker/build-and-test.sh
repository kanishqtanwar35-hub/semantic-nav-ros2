#!/usr/bin/env bash
# Build and test the workspace exactly as CI does.
#
#   docker build -t semantic-nav-dev docker/
#   docker run --rm -v "$PWD:/ws" semantic-nav-dev docker/build-and-test.sh
set -euo pipefail

source /opt/ros/humble/setup.bash

colcon build --symlink-install --event-handlers console_direct+
source install/setup.bash

colcon test --event-handlers console_direct+ --return-code-on-test-failure
colcon test-result --verbose
