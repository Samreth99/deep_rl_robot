#!/usr/bin/env python3
"""
Gazebo environment interface for robot navigation.
Handles interaction with the Gazebo simulator through ROS2.
"""

import time
import math
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from squaternion import Quaternion

from constants import (
    GOAL_PROXIMITY_THRESHOLD, OBSTACLE_DANGER_THRESHOLD, 
    SIMULATION_STEP_DURATION
)
from utils import normalize_angle
from subscribers import OdometrySubscriber, LaserScanSubscriber


class NavigationEnvironment(Node):
    """
    Gazebo environment for robot navigation.
    Interfaces with ROS2 topics and services to control robot and monitor state.
    """
    
    def __init__(self, node_name='navigation_environment'):
        """
        Initialize the navigation environment.
        
        Args:
            node_name (str): Name of the ROS node
        """
        super().__init__(node_name)
        
        # Environment dimensions and state
        self.lidar_points = 360
        self.robot_position = {'x': 0.0, 'y': 0.0}
        
        # Goal configuration
        self.final_destination = {'x': 0.0, 'y': -9.19}
        self.waypoints = [
            {'x': 0.75, 'y': -2.0},   # Waypoint 1
            {'x': 0.75, 'y': -3.0},   # Waypoint 2
            {'x': 0.5, 'y': -4.0},    # Waypoint 3
            {'x': -0.5, 'y': -4.5},   # Waypoint 4
            {'x': -2.0, 'y': -4.5},   # Waypoint 5
            {'x': -2.5, 'y': -5.0},   # Waypoint 6
            {'x': -2.75, 'y': -6.0},  # Waypoint 7
            {'x': -2.0, 'y': -7.5},   # Waypoint 8
            {'x': -1.0, 'y': -8.51}   # Waypoint 9
        ]
        self.current_goal_index = 0
        self.current_goal = self.waypoints[0]
        
        # Robot initial state configuration
        self.robot_initial_state = ModelState()
        self.robot_initial_state.model_name = "r1"
        self.robot_initial_state.pose.position.x = 0.0
        self.robot_initial_state.pose.position.y = 0.0
        self.robot_initial_state.pose.position.z = 0.0
        self.robot_initial_state.pose.orientation.x = 0.0
        self.robot_initial_state.pose.orientation.y = 0.0
        self.robot_initial_state.pose.orientation.z = 0.0
        self.robot_initial_state.pose.orientation.w = 1.0
        
        # Publishers
        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 1)
        self.model_state_publisher = self.create_publisher(ModelState, "gazebo/set_model_state", 10)
        self.goal_marker_publisher = self.create_publisher(MarkerArray, "visualization/goals", 3)
        self.velocity_marker_publisher = self.create_publisher(MarkerArray, "visualization/velocity", 1)
        
        # Service clients
        self.pause_physics_client = self.create_client(Empty, "/pause_physics")
        self.unpause_physics_client = self.create_client(Empty, "/unpause_physics")
        self.reset_world_client = self.create_client(Empty, "/reset_world")
        
        self.odom_subscriber = None
        self.laser_subscriber = None
        self.previous_marker_id = -1
        
        self.get_logger().info("Navigation environment initialized")
    
    def set_subscribers(self, odom_subscriber, laser_subscriber):
        """
        Set the odometry and laser scan subscribers for this environment.
        
        Args:
            odom_subscriber (OdometrySubscriber): Odometry subscriber node
            laser_subscriber (LaserScanSubscriber): Laser scan subscriber node
        """
        self.odom_subscriber = odom_subscriber
        self.laser_subscriber = laser_subscriber
    
    def update_goal(self):
        """
        Update the current goal based on the progress.
        Returns True if the final goal is set.
        """
        if self.current_goal_index < len(self.waypoints):
            self.current_goal = self.waypoints[self.current_goal_index]
            return False
        else:
            self.current_goal = self.final_destination
            return True
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (list): [linear_velocity, angular_velocity]
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        goal_reached = False
        
        # Ensure valid subscribers
        if not self.odom_subscriber or not self.laser_subscriber:
            self.get_logger().error("Subscribers not set - cannot step environment")
            return None, 0, True, {}
        
        # Wait for valid subscriber data
        while (not self.odom_subscriber.has_valid_data() or 
               not self.laser_subscriber.has_valid_data()):
            self.get_logger().info("Waiting for valid sensor data...")
            time.sleep(0.1)
        
        # Send velocity command
        self._send_velocity_command(action)
        self._visualize_state(action)
        
        # Unpause physics for the duration of the step
        self._call_service(self.unpause_physics_client, "unpause physics")
        time.sleep(SIMULATION_STEP_DURATION)
        self._call_service(self.pause_physics_client, "pause physics")
        
        # Get the latest laser scan and odometry data
        laser_data = self.laser_subscriber.get_latest_scan()
        odom_data = self.odom_subscriber.get_latest_odometry()
        
        # Check for collision
        collision_detected, min_distance = self._check_collision(laser_data)
        
        # Update robot position from odometry
        self._update_position_from_odometry(odom_data)
        
        # Calculate state and reward
        distance_to_goal, angle_to_goal = self._calculate_goal_state()
        
        # Check if goal is reached
        if distance_to_goal < GOAL_PROXIMITY_THRESHOLD:
            if self.current_goal_index < len(self.waypoints):
                self.get_logger().info(f"Waypoint {self.current_goal_index + 1} reached!")
                self.current_goal_index += 1
                is_final = self.update_goal()
                self._visualize_goals()
                goal_reached = True
                done = is_final  
            else:
                self.get_logger().info("Final destination reached!")
                goal_reached = True
                done = True
        else:
            done = collision_detected
        
        # Construct state vector
        state = self._construct_state(laser_data, distance_to_goal, angle_to_goal, action)
        
        # Calculate reward
        reward = self._calculate_reward(goal_reached, collision_detected, 
                                       action, distance_to_goal, min_distance)

        info = {
            'goal_reached': goal_reached,
            'collision': collision_detected,
            'distance_to_goal': distance_to_goal,
            'min_obstacle_distance': min_distance
        }
        
        return state, reward, done, info
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial state observation
        """
        # Reset world
        self._call_service(self.reset_world_client, "reset world")
        
        # Reset robot position with random orientation
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        
        self.robot_initial_state.pose.orientation.x = quaternion.x
        self.robot_initial_state.pose.orientation.y = quaternion.y
        self.robot_initial_state.pose.orientation.z = quaternion.z
        self.robot_initial_state.pose.orientation.w = quaternion.w
        
        self.model_state_publisher.publish(self.robot_initial_state)
        
        # Reset goal
        self.current_goal_index = 0
        self.update_goal()
        self._visualize_goals([0.0, 0.0]) 
        
        # Unpause physics briefly to let everything settle
        self._call_service(self.unpause_physics_client, "unpause physics")
        time.sleep(SIMULATION_STEP_DURATION)
        self._call_service(self.pause_physics_client, "pause physics")
        
        # Get initial state
        laser_data = self.laser_subscriber.get_latest_scan()
        self._update_position_from_odometry(self.odom_subscriber.get_latest_odometry())
        
        distance_to_goal, angle_to_goal = self._calculate_goal_state()
        
        # Construct initial state
        state = self._construct_state(laser_data, distance_to_goal, angle_to_goal, [0.0, 0.0])
        
        return state
    
    def _send_velocity_command(self, action):
        """
        Send velocity command to the robot.
        
        Args:
            action (list): [linear_velocity, angular_velocity]
        """
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.velocity_publisher.publish(vel_cmd)
    
    def _check_collision(self, laser_data):
        """
        Check if a collision is detected in the laser scan data.
        
        Args:
            laser_data (np.ndarray): Laser scan data
            
        Returns:
            tuple: (collision_detected, minimum_distance)
        """
        min_distance = np.min(laser_data)
        collision_detected = min_distance < OBSTACLE_DANGER_THRESHOLD
        
        if collision_detected:
            self.get_logger().info(f"Collision detected! Min distance: {min_distance:.3f}m")
            
        return collision_detected, min_distance
    
    def _update_position_from_odometry(self, odom_data):
        """
        Update robot position from odometry data.
        
        Args:
            odom_data (Odometry): Odometry message
        """
        if odom_data:
            self.robot_position['x'] = float(odom_data.pose.pose.position.x)
            self.robot_position['y'] = float(odom_data.pose.pose.position.y)
    
    def _calculate_goal_state(self):
        """
        Calculate distance and angle to current goal.
        
        Returns:
            tuple: (distance_to_goal, angle_to_goal)
        """
        # Calculate distance to goal
        dx = self.current_goal['x'] - self.robot_position['x']
        dy = self.current_goal['y'] - self.robot_position['y']
        distance = np.linalg.norm([dx, dy])
        
        # Extract robot orientation
        odom_data = self.odom_subscriber.get_latest_odometry()
        if not odom_data:
            return distance, 0.0
            
        quaternion = Quaternion(
            odom_data.pose.pose.orientation.w,
            odom_data.pose.pose.orientation.x,
            odom_data.pose.pose.orientation.y,
            odom_data.pose.pose.orientation.z
        )
        euler = quaternion.to_euler(degrees=False)
        robot_angle = euler[2]  
        
        # Calculate angle to goal relative to robot orientation
        goal_angle = math.atan2(dy, dx)
        rel_angle = normalize_angle(goal_angle - robot_angle)
        
        return distance, rel_angle
    
    def _construct_state(self, laser_data, distance_to_goal, angle_to_goal, action):
        """
        Construct the state vector for the agent.
        
        Args:
            laser_data (np.ndarray): Laser scan data
            distance_to_goal (float): Distance to current goal
            angle_to_goal (float): Angle to current goal
            action (list): [linear_velocity, angular_velocity]
            
        Returns:
            np.ndarray: State vector
        """
        # Robot state: [distance_to_goal, angle_to_goal, current_linear_vel, current_angular_vel]
        robot_state = np.array([distance_to_goal, angle_to_goal, action[0], action[1]])
        
        # Combine laser data and robot state
        state = np.concatenate([laser_data, robot_state])
        
        return state
    
    def _calculate_reward(self, goal_reached, collision, action, distance_to_goal, min_obstacle_dist):
        """
        Calculate reward for current state and action.
        
        Args:
            goal_reached (bool): Whether a goal was reached
            collision (bool): Whether a collision occurred
            action (list): [linear_velocity, angular_velocity]
            distance_to_goal (float): Distance to current goal
            min_obstacle_dist (float): Minimum distance to obstacle
            
        Returns:
            float: Calculated reward
        """
        if goal_reached:
            self.get_logger().info("Reward: +100 (Goal reached)")
            return 100.0
        elif collision:
            self.get_logger().info("Reward: -100 (Collision)")
            return -100.0
        else:
            # Base reward for making progress toward goal
            distance_reward = 1.0 - distance_to_goal / 10.0
            
            # Reward for moving forward
            velocity_reward = action[0] / 2.0
            
            # Penalty for being close to obstacles (only if closer than 1m)
            obstacle_penalty = 0.0
            if min_obstacle_dist < 1.0:
                obstacle_penalty = -0.5 * (1.0 - min_obstacle_dist)
            
            # Penalty for excessive rotation
            rotation_penalty = -0.5 * abs(action[1])
            
            total_reward = distance_reward + velocity_reward + obstacle_penalty + rotation_penalty
            
            return total_reward
    
    def _visualize_goals(self, action=None):
        """
        Visualize goals and waypoints in RViz.
        
        Args:
            action (list, optional): Current action for visualization
        """
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.id = self.current_goal_index
        marker.pose.position.x = float(self.current_goal['x'])
        marker.pose.position.y = float(self.current_goal['y'])
        marker_array.markers.append(marker)
        
        self.goal_marker_publisher.publish(marker_array)
        self.previous_marker_id = marker.id
    
    def _visualize_state(self, action):
        """
        Visualize current robot state in RViz.
        
        Args:
            action (list): [linear_velocity, angular_velocity]
        """
        self._visualize_goals()
        
        if action:
            linear_marker_array = MarkerArray()
            linear_marker = Marker()
            linear_marker.header.frame_id = "odom"
            linear_marker.type = Marker.CUBE
            linear_marker.action = Marker.ADD
            linear_marker.scale.x = float(abs(action[0]))
            linear_marker.scale.y = 0.1
            linear_marker.scale.z = 0.01
            linear_marker.color.a = 1.0
            linear_marker.color.r = 1.0
            linear_marker.color.g = 0.0
            linear_marker.color.b = 0.0
            linear_marker.pose.orientation.w = 1.0
            linear_marker.pose.position.x = 5.0
            linear_marker.pose.position.y = 0.0
            linear_marker.pose.position.z = 0.0
            linear_marker_array.markers.append(linear_marker)
            self.velocity_marker_publisher.publish(linear_marker_array)
            
            angular_marker_array = MarkerArray()
            angular_marker = Marker()
            angular_marker.header.frame_id = "odom"
            angular_marker.type = Marker.CUBE
            angular_marker.action = Marker.ADD
            angular_marker.scale.x = float(abs(action[1]))
            angular_marker.scale.y = 0.1
            angular_marker.scale.z = 0.01
            angular_marker.color.a = 1.0
            angular_marker.color.r = 1.0
            angular_marker.color.g = 0.0
            angular_marker.color.b = 0.0
            angular_marker.pose.orientation.w = 1.0
            angular_marker.pose.position.x = 5.0
            angular_marker.pose.position.y = 0.2
            angular_marker.pose.position.z = 0.0
            angular_marker_array.markers.append(angular_marker)
            self.velocity_marker_publisher.publish(angular_marker_array)
    
    def _call_service(self, client, service_name):
        """
        Call a ROS service with error handling.
        
        Args:
            client: ROS service client
            service_name (str): Name of the service for logging
        """
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"{service_name} service not available, waiting...")
            return False
        
        try:
            client.call_async(Empty.Request())
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to call {service_name} service: {str(e)}")
            return False


def create_environment():
    """
    Create and initialize the navigation environment.
    
    Returns:
        NavigationEnvironment: Initialized environment
    """
    return NavigationEnvironment()