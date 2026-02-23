"""
Neural network components for move prediction.

This module implements the SnakeNet neural network for predicting optimal
moves in the Snake game based on extracted game state features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SnakeNet(nn.Module):
    """
    Neural network for Snake Game move prediction.
    
    Architecture: 19 → 200 → 200 → 3
    - Input: 19 game state features
    - Hidden layers: 2 layers of 200 nodes each with ReLU activation
    - Output: 3 move probabilities (Left, Straight, Right) with softmax
    """
    
    def __init__(self, input_features: int = 19, hidden_size: int = 200, output_actions: int = 3):
        """
        Initialize the neural network.
        
        Args:
            input_features: Number of input features (default: 19)
            hidden_size: Size of hidden layers (default: 200)
            output_actions: Number of output actions (default: 3 for L/S/R)
        """
        super(SnakeNet, self).__init__()
        
        # Define network layers
        self.input_layer = nn.Linear(input_features, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_actions)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, output_actions)
        """
        # Ensure input is 2D (batch_size, features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Forward pass through layers
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        
        # Apply softmax to get probabilities
        x = self.softmax(x)
        
        return x
    
    def predict_move(self, features: torch.Tensor) -> Tuple[int, float]:
        """
        Predict the best move from game state features.
        
        Args:
            features: Game state features tensor
            
        Returns:
            Tuple[int, float]: (predicted_action_index, confidence)
                - predicted_action_index: 0=Left, 1=Straight, 2=Right
                - confidence: Probability of the predicted action
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            probabilities = self.forward(features)
            predicted_action = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_action].item()
        
        return predicted_action, confidence
    
    def get_move_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get probabilities for all possible moves.
        
        Args:
            features: Game state features tensor
            
        Returns:
            torch.Tensor: Probabilities for [Left, Straight, Right]
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            probabilities = self.forward(features)
        
        return probabilities.squeeze()
    
    def set_training_mode(self, training: bool = True):
        """
        Set the network to training or evaluation mode.
        
        Args:
            training: If True, set to training mode; if False, set to evaluation mode
        """
        if training:
            self.train()
        else:
            self.eval()
    
    def get_network_info(self) -> dict:
        """
        Get information about the network architecture.
        
        Returns:
            dict: Network architecture information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': '19 → 200 → 200 → 3',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_features': self.input_layer.in_features,
            'hidden_size': self.hidden_layer.in_features,
            'output_actions': self.output_layer.out_features
        }