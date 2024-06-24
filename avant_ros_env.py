from sparse_to_dense_reward.avant_dynamics import AvantDynamics
import rclpy
import rclpy
import torch
import time
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros

SIM_DT = 0.01 # 10ms


class AvantRosEnv(Node):
    def __init__(self):
        super().__init__('avant_ros_env')
        
        self.dynamics = AvantDynamics(SIM_DT, "cpu", True)

        self.odometry_publisher = self.create_publisher(Odometry, 'wheel_odometry', 10)
        self.resolver_publisher = self.create_publisher(JointState, 'resolver', 10)

        self.motion_command_publisher = self.create_subscription(
            JointState, 'motion_commands', self.motion_command_listener_callback, 10
        )
        self.goal_subscription = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_listener_callback, 10
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)


        self.controls = torch.zeros([1, 2], dtype=torch.float32)
        self.states = torch.zeros([1, len(self.dynamics.lbx)], dtype=torch.float32)
        self.goal = torch.zeros(3, dtype=torch.float32)

        self.timer = self.create_timer(SIM_DT, self.sim_loop)

    def motion_command_listener_callback(self, msg):
        self.controls[0, 0] = msg.position[1] * msg.position[0]
        self.controls[0, 1] = msg.position[0] * msg.position[2] 

    def goal_listener_callback(self, msg):
        pose = msg.pose
        position = pose.position
        self.goal[0] = position.x
        self.goal[1] = position.y
        orientation = pose.orientation
        quats = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        self.goal[2] = quats.as_euler('xyz', degrees=False)[2]

    def sim_loop(self):
        t1 = time.time_ns()

        with torch.no_grad():
            self.states = self.dynamics.discrete_dynamics_fun(self.states, self.controls)

        odometry = Odometry()
        odometry.header.stamp = self.get_clock().now().to_msg()
        odometry.header.frame_id = 'odom'
        odometry.child_frame_id = 'base_link'
        odometry.pose.pose.position.x = self.states[0, self.dynamics.x_f_idx].item()
        odometry.pose.pose.position.y = self.states[0, self.dynamics.y_f_idx].item()
        euler = R.from_euler("xyz", [0, 0, self.states[0, self.dynamics.theta_f_idx].item()], degrees=False)
        quats = euler.as_quat()
        odometry.pose.pose.orientation.x = quats[0]
        odometry.pose.pose.orientation.y = quats[1]
        odometry.pose.pose.orientation.z = quats[2]
        odometry.pose.pose.orientation.w = quats[3]
        self.odometry_publisher.publish(odometry)

        resolver_state = JointState()
        resolver_state.position = [self.states[0, self.dynamics.beta_idx].item()]
        resolver_state.velocity = [self.states[0, self.dynamics.dot_beta_idx].item()]
        self.resolver_publisher.publish(resolver_state)

        self.broadcast_tf(odometry)

        t2 = time.time_ns()
        print(f"took {(t2-t1)/1e6} ms")

    def broadcast_tf(self, odometry):
        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'

        transform.transform.translation.x = odometry.pose.pose.position.x
        transform.transform.translation.y = odometry.pose.pose.position.y
        transform.transform.translation.z = odometry.pose.pose.position.z

        transform.transform.rotation = odometry.pose.pose.orientation

        self.tf_broadcaster.sendTransform(transform)

        # Broadcasting static transform for the map frame
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'map'
        static_transform.child_frame_id = 'odom'

        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0

        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(static_transform)

def main(args=None):
	rclpy.init(args=args)
	motion_control = AvantRosEnv()
	rclpy.spin(motion_control)
	
	motion_control.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
    main()