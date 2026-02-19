# CMakeLists.txt for rosetta package
cmake_minimum_required(VERSION 3.16)
project(rosetta)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/episode_recorder_node
  scripts/rosetta_client_node
  scripts/port_bags
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install contract files
install(DIRECTORY
  contracts
  DESTINATION share/${PROJECT_NAME}
)

# Install parameter files
install(DIRECTORY
  params
  DESTINATION share/${PROJECT_NAME}
)

# Enable testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  # Integration tests with launch_testing
  find_package(ament_cmake_ros REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  
  # Function to add isolated launch tests
  function(add_ros_isolated_launch_test path)
    set(RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
    add_launch_test("${path}" RUNNER "${RUNNER}" ${ARGN})
  endfunction()
  
  # Integration tests can be added here when available
  # Example:
  # add_ros_isolated_launch_test(
  #   tests/policy_bridge_launch.test.py
  #   TIMEOUT 60
  # )
endif()

# Package configuration
ament_package()
