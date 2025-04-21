#!/usr/bin/env python3
"""
Main entry point for training the TD3 agent for robot navigation.
Orchestrates the interaction between agent, environment, and ROS2.
"""

import os
import time
import threading
import numpy as np
import torch

import rclpy
from rclpy.executors import MultiThreadedExecutor

from constants import (
    RANDOM_SEED, DEVICE, COMBINED_STATE_DIM, ACTION_DIM, MAX_ACTION_VALUE,
    MAX_TIMESTEPS, MAX_EPISODE_LENGTH, EVAL_FREQUENCY, EVAL_EPISODES,
    INITIAL_EXPLORATION_NOISE, EXPLORATION_NOISE_DECAY_STEPS,
    MINIMUM_EXPLORATION_NOISE, BATCH_SIZE, DISCOUNT_FACTOR, TAU,
    POLICY_NOISE, NOISE_CLIP, POLICY_UPDATE_FREQUENCY,
    MODEL_NAME, RESULTS_DIR, MODELS_DIR
)
from agent import TD3Agent
from environment import create_environment
from subscribers import create_subscribers
from utils import ExperienceReplayBuffer, evaluate_agent, ensure_directories_exist


def main():
    """
    Main function to run the TD3 training for robot navigation.
    """
    rclpy.init()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    ensure_directories_exist([RESULTS_DIR, MODELS_DIR])
    
    # Create environment and subscribers
    env = create_environment()
    odom_subscriber, laser_subscriber = create_subscribers()
    
    # Connect environment with subscribers
    env.set_subscribers(odom_subscriber, laser_subscriber)
    executor = MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(laser_subscriber)
    executor.add_node(env)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    env.get_logger().info("Waiting for initial sensor data...")
    max_wait_time = 10  # seconds
    start_time = time.time()
    
    while (not odom_subscriber.has_valid_data() or 
           not laser_subscriber.has_valid_data()):
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            env.get_logger().warn(f"Timed out waiting for sensor data after {max_wait_time}s")
            break
        time.sleep(0.5)
    
    env.get_logger().info("Sensor data received. Initializing training...")
    
    # Create TD3 agent
    agent = TD3Agent(COMBINED_STATE_DIM, ACTION_DIM, MAX_ACTION_VALUE)
    
    try:
        agent.load(MODEL_NAME, MODELS_DIR)
        env.get_logger().info(f"Loaded existing model from {MODELS_DIR}/{MODEL_NAME}")
    except Exception as e:
        env.get_logger().info(f"Could not load model: {str(e)}. Starting with new model.")
    
    # Create replay buffer
    replay_buffer = ExperienceReplayBuffer(
        buffer_capacity=1e6,
        state_dim=COMBINED_STATE_DIM,
        action_dim=ACTION_DIM,
        seed=RANDOM_SEED
    )
    
    # Initialize tracking variables
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    evaluation_rewards = []
    
    # Current exploration noise (will decay over time)
    exploration_noise = INITIAL_EXPLORATION_NOISE
    
    # Main training loop
    try:
        while total_timesteps < MAX_TIMESTEPS and rclpy.ok():
            # Start a new episode if previous one is done
            if done:
                env.get_logger().info(f"Starting episode {episode_num + 1}")
                
                # Perform evaluation if needed
                if timesteps_since_eval >= EVAL_FREQUENCY:
                    env.get_logger().info(f"Evaluating after {total_timesteps} timesteps...")
                    avg_reward, collision_rate = evaluate_agent(
                        agent=agent,
                        env=env,
                        max_steps=MAX_EPISODE_LENGTH,
                        num_episodes=EVAL_EPISODES
                    )
                    
                    # Record evaluation results
                    evaluation_rewards.append(avg_reward)
                    np.save(f"{RESULTS_DIR}/{MODEL_NAME}_rewards.npy", np.array(evaluation_rewards))
                    
                    # Save model after evaluation
                    agent.save(MODEL_NAME, MODELS_DIR)
                    
                    # Reset evaluation counter
                    timesteps_since_eval = 0
                
                # Reset environment for new episode
                state = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
            
            # Update exploration noise with decay
            if exploration_noise > MINIMUM_EXPLORATION_NOISE:
                decay_amount = (INITIAL_EXPLORATION_NOISE - MINIMUM_EXPLORATION_NOISE) / EXPLORATION_NOISE_DECAY_STEPS
                exploration_noise = max(MINIMUM_EXPLORATION_NOISE, exploration_noise - decay_amount)
            
            # Select action with exploration
            action = agent.select_action(state, add_noise=True, noise_scale=exploration_noise)
            
            # Scale action for environment (linear velocity from [-1,1] to [0,1])
            scaled_action = [(action[0] + 1) / 2, action[1]]
            
            # Execute action in environment
            next_state, reward, done, info = env.step(scaled_action)
            

            done_for_buffer = float(done)
            if episode_timesteps + 1 == MAX_EPISODE_LENGTH:
                done_for_buffer = 0.0
                done = True
            
            replay_buffer.add(state, action, reward, done_for_buffer, next_state)
        
            state = next_state
            episode_reward += reward
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            
            if replay_buffer.size() > BATCH_SIZE:
                metrics = agent.train(replay_buffer, BATCH_SIZE)
                
                # Log training metrics periodically
                if total_timesteps % 1000 == 0:
                    env.get_logger().info(
                        f"Training metrics - "
                        f"Critic Loss: {metrics['critic_loss']:.3f}, "
                        f"Avg Q-value: {metrics['avg_q_value']:.3f}"
                    )
        
            if done:
                env.get_logger().info(
                    f"Episode {episode_num}: {episode_timesteps} steps, "
                    f"reward: {episode_reward:.2f}, "
                    f"exploration: {exploration_noise:.2f}"
                )
    
    except KeyboardInterrupt:
        env.get_logger().info("Training interrupted by user")
    
    # Save final model
    agent.save(MODEL_NAME, MODELS_DIR)
    env.get_logger().info(f"Training complete. Model saved to {MODELS_DIR}/{MODEL_NAME}")

    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()