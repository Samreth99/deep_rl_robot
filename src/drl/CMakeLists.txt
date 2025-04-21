cmake_minimum_required(VERSION 3.8)
project(drl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    launch
    scripts
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
    scripts/replay_buffer.py
    scripts/train_model.py
    scripts/test_model.py
    launch/train_launch.py
    launch/test_launch.py

  DESTINATION lib/${PROJECT_NAME}
)


ament_package()
