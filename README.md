# Deep Reinforcement Learning for Robot Navigation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular implementation of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm for autonomous mobile robot navigation in ROS2 and Gazebo simulation environments.
![Image](https://github.com/user-attachments/assets/619ffd87-c641-4e9a-98e3-27f85f5c5c13)


## Overview

This project implements a reinforcement learning approach to autonomous navigation for mobile robots using the TD3 algorithm. The system enables a differential drive robot equipped with a laser scanner to:

- Navigate through complex environments
- Avoid obstacles using real-time sensor data
- Reach goal positions through waypoint navigation
- Learn optimal policies through interaction with the environment

## Key Features

🤖 **Advanced RL Algorithm**  
Implementation of TD3 with:  
- Twin critic networks to reduce overestimation bias  
- Delayed policy updates for stability  
- Target policy smoothing for robustness  

🚀 **ROS2 Integration**  
- Full compatibility with ROS2 Humble  
- Modular node architecture for scalability  
- Real-time sensor fusion from laser scans and odometry

🗺️ **Simulation Environment**  
- Gazebo physics environment simulation  
- Custom Robocon 2024 competition map (12x12m)  
- Dynamic obstacle avoidance testing

<details>
  <summary>Requirements</summary>

| DEPENDENCY       | VERSION        |
|-------------------|----------------|
| ROS2             | Humble   |
| Gazebo           | 11.0+          |
| Python           | 3.8+           |
| PyTorch          | 1.8+           |
| NumPy            | 1.20+          |
| squaternion      | 0.4.2+         |

</details>

## Installation

1. **Clone repository**
   ```bash
   cd ~/ros2_ws/src/
   git clone https://github.com/yourusername/robot-td3-navigation.git
   ```
2. **Install Python dependencies**
   ```bash
   pip install torch numpy squaternion
   ```
3. **Build ROS2 workspace**
   ```bash
   cd ~/ros2_ws/
   colcon build
   source ~/ros2_ws/install/setup.bash
   ```

## Training
   ```bash
   ros2 launch drl training.launch.py
   ```
## Testing
   ```bash
   ros2 launch drl testing.launch.py
   ```
## Results
  After training for 2390 episodes:
- The agent achieved an average reward of 1053
- Zero collision rate in the evaluation
- Successful navigation through all waypoints to the final destination
- The Q-values converged to around 750 (max) and 360 (average), indicating stable learning.
![Image](https://github.com/user-attachments/assets/82513bf3-bba9-4b39-832a-4fb3c46e414f)
![Image](https://github.com/user-attachments/assets/70b6ebed-628a-4161-b1a1-ec8a8e1011ea)
![Image](https://github.com/user-attachments/assets/3d713dd5-4bb0-4a27-8a88-4684f1b10dda)

For more details, please refer to the documentation.[Thesis.Presentation.pdf](https://github.com/user-attachments/files/19834484/Thesis.Presentation.pdf)
