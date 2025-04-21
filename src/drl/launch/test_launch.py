from launch import LaunchDescription
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription, TimerAction
from launch.substitutions import Command
import os
from ament_index_python.packages import get_package_share_path, get_package_share_directory, get_package_prefix
from os import pathsep
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    urdf_path = os.path.join(get_package_share_path('my_robot_description'), 'urdf', 'bumper.urdf.xacro')
    rviz_config_path = os.path.join(get_package_share_path('my_robot_description'), 'rviz', 'urdf_config.rviz')

    my_robot_description_prefix = get_package_prefix('my_robot_description')
    model_path = os.path.join(get_package_share_directory('my_robot_description'), 'models')
    model_path += pathsep + os.path.join(my_robot_description_prefix, "share")

    env_variable = SetEnvironmentVariable("GAZEBO_MODEL_PATH", model_path)

    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description}]
    )

    world_path = os.path.join(get_package_share_path('my_robot_bringup'), 'worlds', 'victory.world')
    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzserver.launch.py")),
        launch_arguments={'world': world_path}.items()
    )

    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzclient.launch.py"))
    )

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-x","0.0","-y","0.0","-entity", "bumper", "-topic", "robot_description"],
        output = "screen"
    )

    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=['-d', rviz_config_path]
    )

    actions = TimerAction(
        period=0.0,
        actions=[Node(
            package='drl',  
            executable='test.py',
            name='test_model',
            output='screen',
            parameters=[{'use_sim_time': True}],
            prefix='python3 '
        )]
    )

    return LaunchDescription([
        env_variable,
        robot_state_publisher_node,
        rviz2_node,
        start_gazebo_server,
        start_gazebo_client,
        actions,
        spawn_robot
    ])
