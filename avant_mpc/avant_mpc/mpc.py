import numpy as np
import config
import time
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from acados_solver import Solver

class AvantMPC(Node, ABC):
    def __init__(self):
        super().__init__('AvantMPC_node')
        self.odom_subscription = self.create_subscription(
            Odometry, '/odometry/local', self.odom_listener_callback, qos_profile_sensor_data
        )
        self.resolver_subscription = self.create_subscription(
            JointState, '/resolver', self.resolver_listener_callback, qos_profile_sensor_data
        ) 
        self.goal_subscription = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_listener_callback, 10
        )
        self.joint_subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_listener_callback, 10
        )

        self.control_publisher = self.create_publisher(Float64MultiArray, '/motion_controller/commands', 10)
        self.horizon_publisher = self.create_publisher(Path, '/controller/horizon', 10)
        timer_period = 0.02  # seconds (20 Hz)
        self.timer = self.create_timer(timer_period, self.solver_callback)

        self.odom_state = np.zeros(4) # x, y, theta, v
        self.joint_state = 0
        self.joint_velocity = 0
        self.wheel_state = np.zeros(4)

        self.goal_pose = None

    def odom_listener_callback(self, msg):
        pose = msg.pose

        position = pose.pose.position
        orientation = pose.pose.orientation
        quats = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        theta = quats.as_euler('xyz', degrees=False)[2]
        velocity = msg.twist.twist.linear.x

        self.odom_state = np.array([position.x, position.y, theta, velocity])

    def resolver_listener_callback(self, msg):
        self.joint_state = msg.position
        self.joint_velocity = msg.velocity

    def goal_listener_callback(self, msg):
        self.goal_pose = msg.pose

    def joint_listener_callback(self, msg):
        names = ["front_left_wheel_joint", "front_right_wheel_joint", "back_left_wheel_joint", "back_right_wheel_joint"]
        self.wheel_state = np.array([msg.velocity[msg.name.index(name)] for name in names])

    @abstractmethod
    def solver_callback(self):
        pass

    def process_solution(self, state_horizon):
        command = Float64MultiArray()
        beta = state_horizon[1, 3]
        v = state_horizon[1, 5]
        alpha = np.pi + beta
        radii = config.avant_lf / np.tan(alpha/2)
        omega = v * radii
        
        if omega > 0 and abs(radii) > config.avant_d / 2:
            R_l = radii - config.avant_d / 2
            R_r = radii + config.avant_d / 2
        elif omega < 0 and abs(radii) > config.avant_d / 2:
            R_l = radii + config.avant_d / 2
            R_r = radii - config.avant_d / 2
        else:
            R_l = radii
            R_r = radii

        v_l = omega / R_l
        v_r = omega / R_r
        omega_l = v_l / config.avant_r
        omega_r = v_r / config.avant_r

        command.data = [omega_r, omega_l, omega_r, omega_l, state_horizon[1, 4]]
        self.control_publisher.publish(command)

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'  # Adjust based on your coordinate frame

        for state in state_horizon:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = 'map'  # Each pose in the path also has a header

            # Set position
            pose_stamped.pose.position.x = state[0]
            pose_stamped.pose.position.y = state[1]

            r = R.from_euler('z', state[2], degrees=False)
            quat = r.as_quat()
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]

            path_msg.poses.append(pose_stamped)

        self.horizon_publisher.publish(path_msg)
    

class AcadosMPC(AvantMPC):
    def __init__(self):
        super().__init__()
        self.solver = Solver(40, 1/10, 1/5)

    def solver_callback(self):
        t1 = time.time_ns()
        
        if self.goal_pose is None:
            self.get_logger().warn("No goal pose received yet")
            command = Float64MultiArray()
            command.data = [0, 0, 0, 0, 0]
            self.control_publisher.publish(command)
            return

        odom_state = self.odom_state.copy()
        orientation = self.goal_pose.orientation
        quats = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        theta = quats.as_euler('xyz', degrees=False)[2]
        goal = np.array([self.goal_pose.position.x, self.goal_pose.position.y, theta])

        solver_state0 = np.r_[odom_state[:3], self.joint_state, self.joint_velocity, odom_state[3]]
        self.solver.initialize(solver_state0, goal)
        state_horizon, control_horizon, success = self.solver.solve()
        #self.get_logger().info(f"solver success {success}")

        self.process_solution(state_horizon)

        t2 = time.time_ns()
        self.get_logger().info(f"Solving took {(t2-t1)/1e6} ms, {self.joint_state}")