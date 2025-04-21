#!/usr/bin/env python3
import rclpy
from rclpy.node import Node


class MyNode(Node): # MODIFY NAME
    def __init__(self):
        super().__init__("first_node") # MODIFY NAME
        self.get_logger().info("Hello from ROS2")


def main(args=None):
    rclpy.init(args=args)
    node = MyNode() # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()