import numpy as np
import config
import time
import casadi as cs
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from acados_mpc.acados_solver import AcadosSolver
from casadi_mpc.casadi_solver import CasadiSolver
from evotorch_mpc.evotorch_solver import EvoTorchSolver, MPCProblem

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
        timer_period = 0.02  # seconds (50 Hz)
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
        self.solver = AcadosSolver(20, 1/10, 1/5, "critic", True)

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

        self.process_solution(state_horizon)

        t2 = time.time_ns()
        self.get_logger().info(f"Solving took {(t2-t1)/1e6} ms, {self.joint_state}")

import torch
import l4casadi as l4c


def HackyFunction(name,ins,outs):
    Vs = cs.symvar(cs.veccat(*ins))
    # Construct new MX symbols on which the original inputs will depend
    Vns = [cs.MX.sym("V",i.size()) for i in ins]
    res = cs.reverse(ins, Vs, [Vns],{"always_inline":True,"never_inline":False})
    # Substitute the original inputs
    return cs.Function(name,Vns,cs.substitute(outs,Vs,res[0]))

class SymbolicMPC(AvantMPC):
    def __init__(self, linearize=False):
        super().__init__()
        self.linearize = linearize
        model = torch.load("critic").eval()
        if linearize:
            self.l4c_model = l4c.realtime.RealTimeL4CasADi(model, device="cpu")
        else:
            self.l4c_model = l4c.l4casadi.L4CasADi(model, device="cpu")

        # States
        x_f = cs.MX.sym("x_f")
        y_f = cs.MX.sym("y_f")
        theta_f = cs.MX.sym("theta_f")
        beta = cs.MX.sym("beta")
        dot_beta = cs.MX.sym("dot_beta")
        v = cs.MX.sym("v")
        ocp_x = cs.vertcat(x_f, y_f, theta_f, beta, dot_beta, v)

        # Controls
        dot_dot_beta = cs.MX.sym("dot_dot_beta")
        a = cs.MX.sym("a")
        ocp_u = cs.vertcat(dot_dot_beta, a)

        # Params
        x_goal = cs.MX.sym("x_goal")
        y_goal = cs.MX.sym("y_goal")
        theta_goal = cs.MX.sym("y_goal")

        # Terminal Q network:
        neural_cost_state = cs.vertcat(x_f,    y_f,    cs.sin(theta_f),    cs.cos(theta_f), 
                                       x_goal, y_goal, cs.sin(theta_goal), cs.cos(theta_goal), 
                                       beta, dot_beta, v, 
                                       0, 0)
        sym_cost_output = self.l4c_model(neural_cost_state)
        
        if linearize:
            neural_cost_params = self.l4c_model.get_sym_params()
            ocp_p = cs.vertcat(x_goal, y_goal, theta_goal, neural_cost_params)
        else:
            ocp_p = cs.vertcat(x_goal, y_goal, theta_goal)

        # Continuous dynamics:
        alpha = cs.pi + beta
        omega_f = v * config.avant_lf / cs.tan(alpha/2)
        f_expr = cs.vertcat(
            v * cs.cos(theta_f),
            v * cs.sin(theta_f),
            omega_f,
            dot_beta,
            dot_dot_beta,
            a
        )
        f = cs.Function('f', [ocp_x, ocp_u, ocp_p], [f_expr])

        # State constraint bounds
        fake_inf = 1e7
        lbx_vec = np.array([
            -fake_inf, -fake_inf, -fake_inf, -config.avant_max_beta, -config.avant_max_dot_beta,
            config.avant_min_v
        ])
        ubx_vec = np.array([
            fake_inf, fake_inf, fake_inf, config.avant_max_beta, config.avant_max_dot_beta, 
            config.avant_max_v
        ])

        # Control constraint bounds
        lbu_vec = np.array([
            -config.avant_max_dot_dot_beta, -config.avant_max_a
        ])
        ubu_vec = np.array([
            config.avant_max_dot_dot_beta, config.avant_max_a,
        ])

        goal_dist = ((x_f - x_goal)**2 + (y_f - y_goal)**2)
        radius_scaler = cs.exp(-(goal_dist/2)**2)
        goal_heading = radius_scaler * 90 * (1 - cs.cos(theta_goal - theta_f))
        
        cost_expr = (1e-1*dot_dot_beta)**2 + (1e-1*a)**2
        l = cs.Function('l', [ocp_x, ocp_u, ocp_p], [cost_expr])

        # terminal_cost_expr = (x_goal - x_f)**2 + (y_goal - y_f)**2 + goal_heading
        # l_t = cs.Function('l_t', [ocp_x, ocp_p], [terminal_cost_expr])

        if linearize:
            l_t = HackyFunction("l_t", neural_cost_state,)
        else:
            l_t = cs.Function("l_t", [ocp_x, ocp_p], [-sym_cost_output])
        
        self.solver = CasadiSolver(
            N=20, h=1/5, ocp_x=ocp_x, ocp_u=ocp_u, ocp_p=ocp_p, lbx_vec=lbx_vec, ubx_vec=ubx_vec, lbu_vec=lbu_vec, ubu_vec=ubu_vec,
            dynamics_fun=f, cost_fun=l, terminal_cost_fun=l_t,
            #terminal_cost_fun: Function=None, terminal_constraints_fun: Function=None, lbg_terminal_vec: np.array=None, ubg_terminal_vec: np.array=None,
            model_external_shared_lib_dir=self.l4c_model.shared_lib_dir,
            model_external_shared_lib_name=self.l4c_model.name
        )

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

        if self.linearize and self.solver.has_terminal_cost:
            x0 = self.solver.xstar.copy()
            # Replicate the neural cost model inputs for linearization:
            goal_values = np.tile(goal, (self.N+1, 1))
            neural_states = np.c_[x0[:, :2], np.sin(x0[:, 2]), np.cos(x0[:, 2]), 
                                  goal_values[:, :2], np.sin(goal_values[:, 2]), np.cos(goal_values[:, 2]), 
                                  x0[:, 3:], 
                                  np.r_[np.zeros([self.solver.N+1, 2])]]
            neural_cost_params = self.l4c_model.get_params(neural_states)
            params = [np.r_[goal, neural_cost_params[i]] for i in range(self.solver.N + 1)]
        else:
            params = [goal for _ in range(self.solver.N + 1 if self.solver.has_terminal_cost else 0)]

        state_horizon, control_horizon, sol_val = self.solver.solve(solver_state0, params)
        self.process_solution(state_horizon)

        t2 = time.time_ns()
        self.get_logger().info(f"Solving took {(t2-t1)/1e6} ms, {self.joint_state}")


from evotorch_mpc.dynamics import AvantDynamics
from evotorch_mpc.cost import AvantCost


class EvotorchMPC(AvantMPC):
    def __init__(self):
        super().__init__()
        dynamics = AvantDynamics(1/5, "cuda")
        cost = AvantCost()
        problem = MPCProblem(N=10, dynamics=dynamics, cost=cost, device="cuda")
        self.solver = EvoTorchSolver(problem)
    
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
        goal = np.array([self.goal_pose.position.x, self.goal_pose.position.y, theta], dtype=np.float32)
        solver_state0 = np.r_[odom_state[:3], self.joint_state, self.joint_velocity, odom_state[3]]

        solver_state0 = torch.from_numpy(solver_state0).cuda()
        params = torch.from_numpy(goal).cuda()

        state_horizon = self.solver.solve(solver_state0, params)

        self.process_solution(state_horizon)

        t2 = time.time_ns()
        self.get_logger().info(f"Solving took {(t2-t1)/1e6} ms, {self.joint_state}")