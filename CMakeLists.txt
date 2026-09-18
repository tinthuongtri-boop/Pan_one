cmake_minimum_required(VERSION 3.5)
project(Pan_one)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Khai báo các gói phụ thuộc ROS 2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(global_planner_node src/global_planner_node.cpp)
ament_target_dependencies(global_planner_node 
  rclcpp nav_msgs geometry_msgs tf2_ros tf2_geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# 3. Cài đặt file thực thi (Executable) vào hệ thống
install(TARGETS
  global_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

# Cài đặt các thư mục cấu hình hiện có của bạn
install(
  DIRECTORY config description launch worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()