cmake_minimum_required(VERSION 3.5)
project(mono_vo)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# c++ stuff
# include_directories(include ${ament_INCLUDE_DIRS})
# add_executable(mono_vo_node src/mono_vo.cpp)
# ament_target_dependencies(mono_vo_node rclcpp cv_bridge geometry_msgs image_transport sensor_msgs std_msgs tf2 tf2_ros)
# install(TARGETS
#   mono_vo_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# python stuff
# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/mono_vo_node.py
  scripts/mono_vl_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(rclpy std_msgs)

# prevent error message 'file xxx.launch.xml was not found in the share directory of package'
install(DIRECTORY
  launch
  # worlds
  # scripts
  DESTINATION share/${PROJECT_NAME}/
)

# install package.xml, register the package with the ament index,
# and install config/target files for CMake so that it can be found by other packages using find_package
ament_package()
