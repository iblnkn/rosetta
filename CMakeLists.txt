# CMakeLists.txt for rosetta package
cmake_minimum_required(VERSION 3.16)
project(rosetta NONE)

# Enable testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Package configuration
ament_package()
