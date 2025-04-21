#!/usr/bin/env python3
"""
Utility functions and classes for the robot navigation system.
Includes the replay buffer implementation and evaluation functions.
"""

import os
import numpy as np
import torch
from constants import RANDOM_SEED, EVAL_EPISODES


class ExperienceReplayBuffer:
    """
    Replay buffer to store and sample experiences for training.
    Implements uniform random sampling from the buffer.
    """
    
    def __init__(self, buffer_capacity, state_dim, action_dim, seed=RANDOM_SEED):
        """
        Initialize the replay buffer with given capacity.
        
        Args:
            buffer_capacity (int): Maximum number of transitions to store
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            seed (int): Random seed for reproducibility
        """
        self.buffer_capacity = int(buffer_capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.counter = 0
        self.full = False

        np.random.seed(seed)
        
        # Initialize memory buffers
        self.state_buffer = np.zeros((self.buffer_capacity, self.state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_dim), dtype=np.float32)
        self.done_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        
    def add(self, state, action, reward, done, next_state):
        """
        Add a new experience to the buffer.
        
        Args:
            state (np.ndarray): Current state
            action (np.ndarray): Action taken
            reward (float): Reward received
            done (bool): Whether the episode ended
            next_state (np.ndarray): Next state
        """
        # Store experience in the buffer
        index = self.counter % self.buffer_capacity
        
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.done_buffer[index] = done
        self.next_state_buffer[index] = next_state
        
        # Update counter and full flag
        self.counter += 1
        if self.counter >= self.buffer_capacity:
            self.full = True
            
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        # Determine the maximum index for sampling
        max_index = self.buffer_capacity if self.full else self.counter
        
        # Sample random indices
        indices = np.random.randint(0, max_index, size=batch_size)
        
        # Retrieve samples
        state_batch = self.state_buffer[indices]
        action_batch = self.action_buffer[indices]
        reward_batch = self.reward_buffer[indices]
        next_state_batch = self.next_state_buffer[indices]
        done_batch = self.done_buffer[indices]
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def size(self):
        """
        Get the current size of the buffer.
        
        Returns:
            int: Current number of experiences in the buffer
        """
        return self.buffer_capacity if self.full else self.counter


def ensure_directories_exist(directories):
    """
    Ensure that the given directories exist, create them if necessary.
    
    Args:
        directories (list): List of directory paths to check/create
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def evaluate_agent(agent, env, max_steps=500, num_episodes=EVAL_EPISODES):
    """
    Evaluate the agent's performance in the environment.
    
    Args:
        agent: The TD3 agent to evaluate
        env: The environment to evaluate in
        max_steps (int): Maximum steps per episode
        num_episodes (int): Number of episodes to evaluate
        
    Returns:
        float: Average reward per episode
        float: Collision rate
    """
    total_reward = 0.0
    collisions = 0
    
    for episode in range(num_episodes):
        env.get_logger().info(f"Evaluation episode {episode+1}/{num_episodes}")
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Select action without exploration noise
            action = agent.select_action(np.array(state), add_noise=False)
            
            # Convert action range for environment
            scaled_action = [(action[0] + 1) / 2, action[1]]  # Map linear velocity from [-1,1] to [0,1]
            
            # Take step in environment
            next_state, reward, done, info = env.step(scaled_action)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Check for collision
            if reward < -90:
                collisions += 1
        
        total_reward += episode_reward
        env.get_logger().info(f"Episode {episode+1} reward: {episode_reward}")
    
    # Calculate averages
    avg_reward = total_reward / num_episodes
    collision_rate = collisions / num_episodes
    
    env.get_logger().info("===== Evaluation Results =====")
    env.get_logger().info(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    env.get_logger().info(f"Collision Rate: {collision_rate:.2f}")
    env.get_logger().info("=============================")
    
    return avg_reward, collision_rate


def normalize_angle(angle):
    """
    Normalize angle to range [-π, π].
    
    Args:
        angle (float): Angle in radians
        
    Returns:
        float: Normalized angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle