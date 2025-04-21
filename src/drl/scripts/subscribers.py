#!/usr/bin/env python3
"""
ROS2 subscribers for the navigation system.
Handles odometry and laser scan data from the Gazebo simulation.
"""

import numpy as np
from numpy import inf
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class OdometrySubscriber(Node):
    """
    Subscriber node for robot odometry data.
    Captures position and orientation information.
    """
    
    def __init__(self, node_name='odometry_subscriber'):
        """
        Initialize the odometry subscriber.
        
        Args:
            node_name (str): Name of the ROS node
        """
        super().__init__(node_name)
        self.latest_odom = None
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10  
        )
        
        self.get_logger().info("Odometry subscriber initialized")
        
    def odometry_callback(self, odom_msg):
        """
        Callback function for odometry messages.
        
        Args:
            odom_msg (Odometry): Odometry message from ROS
        """
        self.latest_odom = odom_msg
        
    def get_latest_odometry(self):
        """
        Get the latest odometry data.
        
        Returns:
            Odometry: Latest odometry message or None if not available
        """
        return self.latest_odom
    
    def has_valid_data(self):
        """
        Check if valid odometry data is available.
        
        Returns:
            bool: True if valid data is available, False otherwise
        """
        return self.latest_odom is not None
    
    def reset(self):
        """Reset the stored odometry data."""
        self.latest_odom = None


class LaserScanSubscriber(Node):
    """
    Subscriber node for laser scan data.
    Processes raw laser scans for obstacle detection and navigation.
    """
    
    def __init__(self, node_name='laser_scan_subscriber'):
        """
        Initialize the laser scan subscriber.
        
        Args:
            node_name (str): Name of the ROS node
        """
        super().__init__(node_name)
        self.latest_scan = None
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_scan_callback,
            10 
        )
        
        self.get_logger().info("Laser scan subscriber initialized")
        
    def laser_scan_callback(self, scan_msg):
        """
        Callback function for laser scan messages.
        Processes the scan to handle infinite values.
        
        Args:
            scan_msg (LaserScan): Laser scan message from ROS
        """
        scan_data = np.array(scan_msg.ranges)
        
        # Replace infinite values with a large fixed value (10 meters)
        scan_data = np.where(scan_data == inf, 10.0, scan_data)
        self.latest_scan = scan_data
        
    def get_latest_scan(self):
        """
        Get the latest processed laser scan data.
        
        Returns:
            np.ndarray: Latest laser scan array or None if not available
        """
        return self.latest_scan
    
    def has_valid_data(self):
        """
        Check if valid laser scan data is available.
        
        Returns:
            bool: True if valid data is available, False otherwise
        """
        return self.latest_scan is not None
    
    def get_min_distance(self):
        """
        Get the minimum distance from the laser scan.
        
        Returns:
            float: Minimum distance value or None if data not available
        """
        if self.latest_scan is not None:
            return np.min(self.latest_scan)
        return None
    
    def reset(self):
        """Reset the stored laser scan data."""
        self.latest_scan = None


def create_subscribers():
    """
    Create and initialize odometry and laser scan subscribers.
    
    Returns:
        tuple: (OdometrySubscriber, LaserScanSubscriber)
    """
    odom_subscriber = OdometrySubscriber()
    laser_subscriber = LaserScanSubscriber()
    
    return odom_subscriber, laser_subscriber