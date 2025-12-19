#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VelocityOverlay: Visualize robot velocity commands overlayed on camera image.

Subscribes to:
  - Image topic (default: /camera/color/image_raw)
  - TwistStamped topic (default: /motion_control/speed_controller/output_cmd)

Publishes:
  - Overlayed image with velocity visualization
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import cv2


class PolicyVisualizer(Node):
    """Node to visualize model output on camera images."""

    def __init__(self):
        super().__init__('policy_visualizer')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('twist_topic', '/motion_control/speed_controller/reference_cmd')
        self.declare_parameter('output_topic', '/policy_visualizer/image')
        self.declare_parameter('linear_min', -1.0)
        self.declare_parameter('linear_max', 2.0)
        self.declare_parameter('angular_min', -0.75)
        self.declare_parameter('angular_max', 0.75)

        # Get parameters
        image_topic = self.get_parameter('image_topic').value
        twist_topic = self.get_parameter('twist_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.linear_min = self.get_parameter('linear_min').value
        self.linear_max = self.get_parameter('linear_max').value
        self.angular_min = self.get_parameter('angular_min').value
        self.angular_max = self.get_parameter('angular_max').value

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Latest twist message
        self.latest_twist = None
        self.twist_lock = False

        # QoS profiles
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile
        )

        self.twist_sub = self.create_subscription(
            TwistStamped,
            twist_topic,
            self.twist_callback,
            qos_profile
        )

        # Publisher
        self.image_pub = self.create_publisher(
            Image,
            output_topic,
            qos_profile
        )

        self.get_logger().info(f'Velocity Overlay Node started')
        self.get_logger().info(f'  Image topic: {image_topic}')
        self.get_logger().info(f'  Twist topic: {twist_topic}')
        self.get_logger().info(f'  Output topic: {output_topic}')
        self.get_logger().info(f'  Linear range: [{self.linear_min}, {self.linear_max}]')
        self.get_logger().info(f'  Angular range: [{self.angular_min}, {self.angular_max}]')

    def twist_callback(self, msg: TwistStamped):
        """Store the latest twist message."""
        self.latest_twist = msg

    def image_callback(self, msg: Image):
        """Process image and overlay velocity visualization."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            
            # Create overlay
            overlay_image = self.create_overlay(cv_image)

            # import cv2
            # cv2.imshow("Overlay", overlay_image)
            # cv2.waitKey(1)
            
            # Convert back to ROS Image
            output_msg = self.bridge.cv2_to_imgmsg(overlay_image, encoding='bgr8')
            output_msg.header = msg.header
            
            # Publish
            self.image_pub.publish(output_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def create_overlay(self, image: np.ndarray) -> np.ndarray:
        """Create velocity overlay on image."""
        overlay = image.copy()
        height, width = overlay.shape[:2]
        
        if self.latest_twist is None:
            # Draw "No Data" message
            cv2.putText(
                overlay,
                "No velocity data",
                (width - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            return overlay
        
        linear_x = self.latest_twist.twist.linear.x
        angular_z = self.latest_twist.twist.angular.z
        
        # Define overlay dimensions
        bar_width = 40
        bar_margin = 10
        bar_x = width - bar_width - bar_margin
        bar_y_start = 50
        bar_height = height - 100
        
        # Draw linear velocity bar (vertical bar on the right)
        self._draw_linear_bar(
            overlay, 
            linear_x, 
            bar_x, 
            bar_y_start, 
            bar_width, 
            bar_height
        )
        
        # Draw angular indicator (line from bottom)
        self._draw_angular_indicator(
            overlay,
            angular_z,
            height,
            width
        )
        
        return overlay

    def _draw_linear_bar(
        self, 
        image: np.ndarray, 
        linear_x: float,
        bar_x: int,
        bar_y: int,
        bar_width: int,
        bar_height: int
    ):
        """Draw vertical bar showing linear velocity."""
        # Draw background bar
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )
        
        # Draw border
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (255, 255, 255),
            2
        )
        
        # Normalize linear velocity to bar height
        normalized = (linear_x - self.linear_min) / (self.linear_max - self.linear_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Calculate fill height (from bottom)
        fill_height = int(normalized * bar_height)
        fill_y = bar_y + bar_height - fill_height
        
        # Choose color based on value
        if linear_x < 0:
            color = (0, 0, 255)  # Red for reverse
        elif linear_x > 0:
            color = (0, 255, 0)  # Green for forward
        else:
            color = (128, 128, 128)  # Gray for zero
        
        # Draw filled portion
        if fill_height > 0:
            cv2.rectangle(
                image,
                (bar_x + 2, fill_y),
                (bar_x + bar_width - 2, bar_y + bar_height - 2),
                color,
                -1
            )
        
        # Draw zero line (where velocity = 0)
        zero_normalized = (0 - self.linear_min) / (self.linear_max - self.linear_min)
        zero_y = bar_y + bar_height - int(zero_normalized * bar_height)
        cv2.line(
            image,
            (bar_x, zero_y),
            (bar_x + bar_width, zero_y),
            (255, 255, 0),
            2
        )
        
        # Draw text with value
        text = f"{linear_x:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = bar_x + (bar_width - text_size[0]) // 2
        text_y = bar_y - 10
        
        # Background for text
        cv2.rectangle(
            image,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        # Label
        cv2.putText(
            image,
            "Linear",
            (bar_x - 10, bar_y + bar_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

    def _draw_angular_indicator(
        self,
        image: np.ndarray,
        angular_z: float,
        height: int,
        width: int
    ):
        """Draw angular velocity indicator as a line from bottom."""
        # Define indicator position and size
        indicator_length = 80
        indicator_base_y = height - 30
        indicator_center_x = width // 2
        
        # Normalize angular velocity to angle
        # angular_z = 0 should point straight up (90 degrees from horizontal)
        # angular_z > 0 should point left (> 90 degrees)
        # angular_z < 0 should point right (< 90 degrees)
        normalized = angular_z / self.angular_max
        normalized = np.clip(normalized, -1.0, 1.0)
        
        # Convert to angle (in radians)
        # 0 angular -> 90 degrees (pi/2)
        # positive angular (left turn) -> angle > 90 degrees
        # negative angular (right turn) -> angle < 90 degrees
        angle_offset = normalized * (np.pi / 3)  # +/- 60 degrees range
        angle_rad = (np.pi / 2) + angle_offset  # Add because positive angular means turn left
        
        # Calculate end point of indicator line
        end_x = int(indicator_center_x + indicator_length * np.cos(angle_rad))
        end_y = int(indicator_base_y - indicator_length * np.sin(angle_rad))
        
        # Draw background circle
        cv2.circle(
            image,
            (indicator_center_x, indicator_base_y),
            10,
            (50, 50, 50),
            -1
        )
        
        # Draw border circle
        cv2.circle(
            image,
            (indicator_center_x, indicator_base_y),
            10,
            (255, 255, 255),
            2
        )
        
        # Choose color based on direction
        if angular_z > 0.05:
            color = (0, 165, 255)  # Orange for left
        elif angular_z < -0.05:
            color = (255, 0, 255)  # Magenta for right
        else:
            color = (0, 255, 255)  # Yellow for straight
        
        # Draw indicator line
        cv2.line(
            image,
            (indicator_center_x, indicator_base_y),
            (end_x, end_y),
            color,
            4
        )
        
        # Draw arrowhead
        cv2.circle(
            image,
            (end_x, end_y),
            6,
            color,
            -1
        )
        
        # Draw zero reference line (straight up)
        ref_end_x = indicator_center_x
        ref_end_y = indicator_base_y - indicator_length
        cv2.line(
            image,
            (indicator_center_x, indicator_base_y),
            (ref_end_x, ref_end_y),
            (128, 128, 128),
            1
        )
        
        # Draw text with value
        text = f"{angular_z:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = indicator_center_x - text_size[0] // 2
        text_y = indicator_base_y + 30
        
        # Background for text
        cv2.rectangle(
            image,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        # Label
        cv2.putText(
            image,
            "Angular",
            (indicator_center_x - 30, text_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = PolicyVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

