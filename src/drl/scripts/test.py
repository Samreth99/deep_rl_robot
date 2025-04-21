#!/usr/bin/env python3
"""
Test script for running a trained TD3 agent in the Gazebo environment.
This script loads a pre-trained model and runs it in the environment without training.
"""

import os
import time
import threading
import numpy as np
import torch
import rclpy
from rclpy.executors import MultiThreadedExecutor

from test_constants import (
    DEVICE, COMBINED_STATE_DIM, ACTION_DIM, 
    MAX_EPISODE_LENGTH, RANDOM_SEED
)
from models import PolicyNetwork
from environment import NavigationEnvironment
from subscribers import OdometrySubscriber, LaserScanSubscriber


class InferenceAgent:
    """
    Simplified agent for inference that only uses the actor network.
    """
    
    def __init__(self, state_dim, action_dim):
        """
        Initialize the inference agent with just the policy network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
        """
        # Initialize just the actor (policy) network for inference
        self.actor = PolicyNetwork(state_dim, action_dim).to(DEVICE)
        
    def get_action(self, state):
        """
        Get action from the policy network.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            np.ndarray: Selected action
        """
        # Convert state to tensor and get action from actor network
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
            
        return action
    
    def load(self, filename, directory):
        """
        Load only the actor model from disk.
        
        Args:
            filename (str): Base name for saved model file
            directory (str): Directory containing the model
        """
        model_path = f"{directory}/{filename}_actor.pth"
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} does not exist")
            
        self.actor.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")


class TestEnvironment(NavigationEnvironment):
    """
    Modified navigation environment for testing with test-specific goals.
    """
    
    def __init__(self, node_name='test_environment'):
        """Initialize the test environment with different goal configuration."""
        super().__init__(node_name)
        
        # Override the final destination and waypoints for testing
        self.final_destination = {'x': -2.75, 'y': -6.0}
        self.waypoints = [
            {'x': 0.75, 'y': -2.0},   # Waypoint 0
            {'x': 0.75, 'y': -3.0},   # Waypoint 1
            {'x': 0.5, 'y': -4.0},    # Waypoint 2
            {'x': -0.5, 'y': -4.5},   # Waypoint 3
            {'x': -2.0, 'y': -4.5},   # Waypoint 4
            {'x': -2.5, 'y': -5.0}    # Waypoint 5
        ]
        self.current_goal_index = 0
        self.update_goal()
        
        self.get_logger().info("Test environment initialized with test-specific goals")


def main():
    """
    Main function to run the trained TD3 agent in test mode.
    """
    # Initialize ROS2
    rclpy.init(args=None)
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Configuration
    file_name = "td3_laserscan"
    models_dir = "./pytorch_models"
    
    # Create environment and subscribers
    env = TestEnvironment()
    odom_subscriber = OdometrySubscriber()
    laser_subscriber = LaserScanSubscriber()
    
    # Connect environment with subscribers
    env.set_subscribers(odom_subscriber, laser_subscriber)
    
    # Setup ROS2 executor for subscribers
    executor = MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(laser_subscriber)
    executor.add_node(env)
    
    # Start executor in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    # Create inference agent
    agent = InferenceAgent(COMBINED_STATE_DIM, ACTION_DIM)
    
    # Load the trained model
    try:
        agent.load(file_name, models_dir)
    except Exception as e:
        env.get_logger().error(f"Failed to load model: {str(e)}")
        rclpy.shutdown()
        return
    
    # Initialize testing variables
    done = True
    episode_timesteps = 0
    
    # Main testing loop
    try:
        env.get_logger().info("Beginning test loop")
        while rclpy.ok():
            # Reset environment at the start or when done
            if done:
                env.get_logger().info("Resetting environment")
                state = env.reset()
                done = False
                episode_timesteps = 0
                
            # Get action from policy
            action = agent.get_action(np.array(state))
            
            # Scale action for environment (linear velocity from [-1,1] to [0,1])
            scaled_action = [(action[0] + 1) / 2, action[1]]
            
            # Execute action in environment
            next_state, reward, done, info = env.step(scaled_action)
            
            # Check if episode ended due to time limit
            if episode_timesteps + 1 == MAX_EPISODE_LENGTH:
                env.get_logger().info("Episode ended due to time limit")
                done = True
            
            # Update state and counters
            state = next_state
            episode_timesteps += 1
            
            # Log episode progress periodically
            if episode_timesteps % 20 == 0:
                env.get_logger().info(f"Step {episode_timesteps}")
    
    except KeyboardInterrupt:
        env.get_logger().info("Testing interrupted by user")
    finally:
        # Shutdown ROS
        rclpy.shutdown()
        executor_thread.join()
        env.get_logger().info("Testing complete")


if __name__ == "__main__":
    main()