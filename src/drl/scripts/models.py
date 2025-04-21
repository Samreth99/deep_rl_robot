#!/usr/bin/env python3
"""
Neural network model architectures for the TD3 agent.
Includes the PolicyNetwork (Actor) and ValueNetwork (Critic) implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import DEVICE, ACTOR_HIDDEN_1, ACTOR_HIDDEN_2, CRITIC_HIDDEN_1, CRITIC_HIDDEN_2


class PolicyNetwork(nn.Module):
    """
    Actor network that maps states to continuous actions.
    Uses tanh activation to bound actions between -1 and 1.
    """
    
    def __init__(self, state_dimension, action_dimension):
        """
        Initialize the policy network architecture.
        
        Args:
            state_dimension (int): Dimension of the state space
            action_dimension (int): Dimension of the action space
        """
        super(PolicyNetwork, self).__init__()
        
        # Network layers
        self.input_layer = nn.Linear(state_dimension, ACTOR_HIDDEN_1)
        self.hidden_layer = nn.Linear(ACTOR_HIDDEN_1, ACTOR_HIDDEN_2)
        self.output_layer = nn.Linear(ACTOR_HIDDEN_2, action_dimension)
        
        # Initialize weights
        self._initialize_parameters()
        
    def forward(self, state):
        """
        Forward pass through the policy network.
        
        Args:
            state (torch.Tensor): Current state tensor
            
        Returns:
            torch.Tensor: Action values bounded by tanh between -1 and 1
        """
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        actions = torch.tanh(self.output_layer(x))
        return actions
    
    def _initialize_parameters(self):
        """Initialize network parameters with uniform distribution."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # Initialize weights with uniform distribution
                nn.init.uniform_(layer.weight, -0.1, 0.1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


class ValueNetwork(nn.Module):
    """
    Critic network that maps state-action pairs to Q-values.
    Implements the twin Q-networks as described in TD3 paper.
    """
    
    def __init__(self, state_dimension, action_dimension):
        """
        Initialize the twin critic networks.
        
        Args:
            state_dimension (int): Dimension of the state space
            action_dimension (int): Dimension of the action space
        """
        super(ValueNetwork, self).__init__()
        
        # First Q-network
        self.q1_input_layer = nn.Linear(state_dimension + action_dimension, CRITIC_HIDDEN_1)
        self.q1_hidden_layer = nn.Linear(CRITIC_HIDDEN_1, CRITIC_HIDDEN_2)
        self.q1_output_layer = nn.Linear(CRITIC_HIDDEN_2, 1)
        
        # Second Q-network
        self.q2_input_layer = nn.Linear(state_dimension + action_dimension, CRITIC_HIDDEN_1)
        self.q2_hidden_layer = nn.Linear(CRITIC_HIDDEN_1, CRITIC_HIDDEN_2)
        self.q2_output_layer = nn.Linear(CRITIC_HIDDEN_2, 1)
        
        # Initialize weights
        self._initialize_parameters()
        
    def forward(self, state, action):
        """
        Forward pass through both Q-networks.
        
        Args:
            state (torch.Tensor): Current state tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            tuple(torch.Tensor, torch.Tensor): Q-values from both networks
        """
        # Concatenate state and action
        sa_concat = torch.cat([state, action], dim=1)
        
        # First Q-network
        q1 = F.relu(self.q1_input_layer(sa_concat))
        q1 = F.relu(self.q1_hidden_layer(q1))
        q1_value = self.q1_output_layer(q1)
        
        # Second Q-network
        q2 = F.relu(self.q2_input_layer(sa_concat))
        q2 = F.relu(self.q2_hidden_layer(q2))
        q2_value = self.q2_output_layer(q2)
        
        return q1_value, q2_value
    
    def q1(self, state, action):
        """
        Get Q-value only from the first Q-network.
        Used for deterministic policy gradient.
        
        Args:
            state (torch.Tensor): Current state tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            torch.Tensor: Q-value from the first network
        """
        sa_concat = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_input_layer(sa_concat))
        q1 = F.relu(self.q1_hidden_layer(q1))
        q1_value = self.q1_output_layer(q1)
        return q1_value
    
    def _initialize_parameters(self):
        """Initialize network parameters with uniform distribution."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.1, 0.1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)