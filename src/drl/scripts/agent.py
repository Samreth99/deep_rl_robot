#!/usr/bin/env python3
"""
Implementation of the Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
Handles training, action selection, and model management.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import PolicyNetwork, ValueNetwork
from constants import (
    DEVICE, DISCOUNT_FACTOR, TAU, POLICY_NOISE, NOISE_CLIP, 
    POLICY_UPDATE_FREQUENCY, MAX_ACTION_VALUE, LOGS_DIR
)


class TD3Agent:
    """
    TD3 Agent implementation with actor-critic architecture.
    Implements the TD3 algorithm with twin critics and delayed policy updates.
    """
    
    def __init__(self, state_dim, action_dim, max_action=MAX_ACTION_VALUE):
        """
        Initialize the TD3 agent with policy and value networks.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            max_action (float): Maximum action value magnitude
        """
        # Initialize actor (policy) networks
        self.actor = PolicyNetwork(state_dim, action_dim).to(DEVICE)
        self.actor_target = PolicyNetwork(state_dim, action_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # Initialize critic (value) networks
        self.critic = ValueNetwork(state_dim, action_dim).to(DEVICE)
        self.critic_target = ValueNetwork(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Other parameters
        self.max_action = max_action
        self.training_iterations = 0
        
        self.writer = SummaryWriter(log_dir=LOGS_DIR)
        
    def select_action(self, state, add_noise=False, noise_scale=0.1):
        """
        Select an action given the current state using the policy network.
        
        Args:
            state (np.ndarray): Current state observation
            add_noise (bool): Whether to add exploration noise
            noise_scale (float): Scale of the exploration noise
            
        Returns:
            np.ndarray: Selected action
        """
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
    
        if add_noise:
            action = action + np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
            
        return action
    
    def train(self, replay_buffer, batch_size=100):
        """
        Train the agent using samples from the replay buffer.
        
        Args:
            replay_buffer: Buffer containing experience transitions
            batch_size (int): Number of samples to use for training
            
        Returns:
            dict: Training metrics
        """
        self.training_iterations += 1
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state_batch).to(DEVICE)
        action = torch.FloatTensor(action_batch).to(DEVICE)
        reward = torch.FloatTensor(reward_batch).to(DEVICE)
        next_state = torch.FloatTensor(next_state_batch).to(DEVICE)
        done = torch.FloatTensor(done_batch).to(DEVICE)
        
        avg_q_value = 0
        max_q_value = float('-inf')
        critic_loss_value = 0
        
        with torch.no_grad():
            noise = torch.randn_like(action) * POLICY_NOISE
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Get target Q-values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)  # Take minimum to reduce overestimation
            
            # Compute target value with Bellman equation
            target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * DISCOUNT_FACTOR * target_q
            
            # Record metrics
            avg_q_value = target_q.mean().item()
            max_q_value = target_q.max().item()
        
        # Update critic networks
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        critic_loss_value = critic_loss.item()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        actor_loss_value = 0
        if self.training_iterations % POLICY_UPDATE_FREQUENCY == 0:
            # Update actor network through deterministic policy gradient
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            actor_loss_value = actor_loss.item()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update_targets()
        
        self.writer.add_scalar("Loss/Critic", critic_loss_value, self.training_iterations)
        self.writer.add_scalar("Q-values/Average", avg_q_value, self.training_iterations)
        self.writer.add_scalar("Q-values/Maximum", max_q_value, self.training_iterations)
        
        if self.training_iterations % POLICY_UPDATE_FREQUENCY == 0:
            self.writer.add_scalar("Loss/Actor", actor_loss_value, self.training_iterations)
            
        return {
            "critic_loss": critic_loss_value,
            "actor_loss": actor_loss_value,
            "avg_q_value": avg_q_value,
            "max_q_value": max_q_value
        }
    
    def _soft_update_targets(self):
        """
        Perform soft update on target networks.
        θ_target = τ*θ_local + (1-τ)*θ_target
        """
        # Update actor target
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    
    def save(self, filename, directory):
        """
        Save model parameters to disk.
        
        Args:
            filename (str): Base name for saved model files
            directory (str): Directory to save models in
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        
    def load(self, filename, directory):
        """
        Load model parameters from disk.
        
        Args:
            filename (str): Base name for saved model files
            directory (str): Directory to load models from
        """
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())