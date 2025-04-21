#!/usr/bin/env python3
"""
Configuration constants for the autonomous navigation system.
Contains parameters for both the environment and the TD3 algorithm.
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
COMBINED_STATE_DIM = LIDAR_SCAN_POINTS + ROBOT_STATE_DIM  

# Action dimensions
ACTION_DIM = 2  # [linear_velocity, angular_velocity]
MAX_ACTION_VALUE = 1.0  

# Neural network constants
ACTOR_HIDDEN_1 = 400  
CRITIC_HIDDEN_1 = 400  
CRITIC_HIDDEN_2 = 300  

# TD3 algorithm parameters
INITIAL_EXPLORATION_NOISE = 1.0 
EXPLORATION_NOISE_DECAY_STEPS = 500000  
MINIMUM_EXPLORATION_NOISE = 0.1 

POLICY_NOISE = 0.2  
NOISE_CLIP = 0.5 
DISCOUNT_FACTOR = 0.99  
TAU = 0.005  
POLICY_UPDATE_FREQUENCY = 2  
BATCH_SIZE = 100  
BUFFER_SIZE = int(1e6)  

# Training settings
MAX_TIMESTEPS = int(5e6)  
MAX_EPISODE_LENGTH = 500  
EVAL_FREQUENCY = int(10e3)  
EVAL_EPISODES = 10 

# File paths
MODEL_NAME = "td3_navigator"  
RESULTS_DIR = "./nav_training/results"
MODELS_DIR = "./nav_training/models"
LOGS_DIR = "./nav_training/logs"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed for reproducibility
RANDOM_SEED = 42