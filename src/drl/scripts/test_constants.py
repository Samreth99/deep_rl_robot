#!/usr/bin/env python3
"""
Configuration constants for the test mode of the autonomous navigation system.
Contains parameters specific for inference/testing rather than training.
"""

import torch
import numpy as np

# Environment constants
GOAL_PROXIMITY_THRESHOLD = 0.36  
OBSTACLE_DANGER_THRESHOLD = 0.2  
SIMULATION_STEP_DURATION = 0.2  

# Environment dimensions
LIDAR_SCAN_POINTS = 360  
ROBOT_STATE_DIM = 4      # [distance_to_goal, angle_to_goal, linear_vel, angular_vel]
COMBINED_STATE_DIM = LIDAR_SCAN_POINTS + ROBOT_STATE_DIM  # Total state dimension

# Action dimensions
ACTION_DIM = 2  # [linear_velocity, angular_velocity]
MAX_ACTION_VALUE = 1.0  

# Neural network constants
ACTOR_HIDDEN_1 = 400  
ACTOR_HIDDEN_2 = 300  

# Testing settings
MAX_EPISODE_LENGTH = 500  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 0