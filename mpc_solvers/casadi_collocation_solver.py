import subprocess
from typing import List
import numpy as np
from casadi import *
from mpc_solvers.mpc_problem import SymbolicMPCProblem


class CasadiCollocationSolver:
    def __init__(self, problem: SymbolicMPCProblem):
        self.problem = problem
        self.N = problem.N
        self.has_terminal_cost = problem.terminal_cost_fun is not None
        self.xstar = None
        self.ustar = None

        self.n_x = problem.ocp_x.size1()
        self.n_u = problem.ocp_u.size1()
        self.n_p = problem.ocp_p.size1()
        self.n_t_p = problem.terminal_ocp_p.size1()
        self.n_terminal_cost_p = problem.terminal_cost_params.size1() if problem.terminal_cost_params is not None else 0

        # Degree of interpolating polynomial
        self.polydeg = 3
        # Get collocation points
        tau_root = np.append(0, collocation_points(self.polydeg, 'legendre'))
        # Coefficients of the collocation equation
        C = np.zeros((self.polydeg + 1, self.polydeg + 1))
        # Coefficients of the continuity equation
        D = np.zeros(self.polydeg + 1)
        # Coefficients of the quadrature function
        B = np.zeros(self.polydeg + 1)
        
        # Construct polynomial basis
        for j in range(self.polydeg + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            poly = np.poly1d([1])
            for r in range(self.polydeg + 1):
                if r != j:
                    poly *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = poly(1.0)
            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(poly)
            for r in range(self.polydeg + 1):
                C[j, r] = pder(tau_root[r])
            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(poly)
            B[j] = pint(1.0)

        # Start with an empty NLP
        w = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        p = []

        # For extracting x and u given w
        x_solution = []
        u_solution = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_x)
        w.append(Xk)
        lbw.append(np.zeros(self.n_x))
        ubw.append(np.zeros(self.n_x))
        x_solution.append(Xk)

        # Formulate the NLP
        for k in range(self.N):
            # New NLP parameter for this stage
            Pk = MX.sym("P_" + str(k), self.n_p)
            p.append(Pk)

            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), self.n_u)
            w.append(Uk)
            lbw.append(problem.lbu_vec)
            ubw.append(problem.ubu_vec)
            u_solution.append(Uk)

            # State at collocation points
            Xc = []
            for j in range(self.polydeg):
                Xkj = MX.sym('X_' + str(k) + '_' + str(j), self.n_x)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(problem.lbx_vec)
                ubw.append(problem.ubx_vec)

            # Loop over collocation points
            Xk_end = D[0] * Xk
            for j in range(1, self.polydeg + 1):
                xp = C[0, j] * Xk
                for r in range(self.polydeg):
                    xp = xp + C[r + 1, j] * Xc[r]
                # dynamics:
                fj = problem.dynamics_fun(Xc[j - 1], Uk, Pk)
                g.append(problem.h * fj - xp)
                lbg.append(np.zeros(self.n_x))
                ubg.append(np.zeros(self.n_x))
                # cost
                qj = problem.cost_fun(Xc[j - 1], Uk, Pk)
                J = J + B[j] * qj * problem.h

                if problem.g_fun is not None and problem.lbg_vec is not None and problem.ubg_vec is not None:
                    # constrain:
                    gj = problem.g_fun(Xc[j - 1], Uk, Pk)
                    g.append(gj)
                    lbg.append(problem.lbg_vec)
                    ubg.append(problem.ubg_vec)
                
                # propagate:
                Xk_end = Xk_end + D[j] * Xc[j - 1]

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), self.n_x)
            w.append(Xk)
            lbw.append(problem.lbx_vec)
            ubw.append(problem.ubx_vec)
            x_solution.append(Xk)

            # Add equality constraint
            g.append(Xk_end - Xk)
            lbg.append(np.zeros(self.n_x))
            ubg.append(np.zeros(self.n_x))

            if problem.g_fun is not None and problem.lbg_vec is not None and problem.ubg_vec is not None:
                # constrain:
                if k != self.N-1:
                    ge = problem.g_fun(Xk, Uk, Pk)
                    g.append(ge)
                    lbg.append(problem.lbg_vec)
                    ubg.append(problem.ubg_vec)

        if problem.terminal_cost_fun is not None:
            # New NLP parameter for terminal stage
            Pk = MX.sym("P_" + str(k+1), self.n_t_p)
            p.append(Pk)
            if problem.ocp_x_to_terminal_state_fun is not None:
                neural_cost_inputs = problem.ocp_x_to_terminal_state_fun(Xk, Pk)
                if self.n_terminal_cost_p > 0:
                    # New NLP parameter for terminal cost linearization params
                    Plk = MX.sym("Pl_" + str(k+1), self.n_terminal_cost_p)
                    p.append(Plk)
                    J = J + problem.terminal_cost_fun(neural_cost_inputs, Plk, Pk)
                else:
                    J = J + problem.terminal_cost_fun(neural_cost_inputs, Pk)
            else:
                J = J + problem.terminal_cost_fun(Xk, Pk)
            
        if problem.terminal_g_fun is not None and problem.terminal_lbg_vec is not None and problem.terminal_ubg_vec is not None:
            ge = problem.terminal_g_fun(Xk, Pk)
            g.append(ge)
            lbg.append(problem.terminal_lbg_vec)
            ubg.append(problem.terminal_ubg_vec)

        # Concatenate vectors
        w = vertcat(*w)
        g = vertcat(*g)
        p = vertcat(*p)
        x_solution = horzcat(*x_solution)
        u_solution = horzcat(*u_solution)
        self.lbw = np.concatenate(lbw)
        self.ubw = np.concatenate(ubw)
        self.lbg = np.concatenate(lbg)
        self.ubg = np.concatenate(ubg)

        # Create an NLP solver
        prob = {'f': J, 'x': w, 'g': g, 'p': p}
        opts = {'ipopt.max_iter': 100, 'ipopt.print_level': 0}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        
        # Generate C code:
        gen_opts = {}
        solver.generate_dependencies("nlp.c", gen_opts)
        if problem.model_external_shared_lib_name is not None and problem.model_external_shared_lib_name is not None:
            # Need to link the l4casadi symbols:
            subprocess.Popen(f"gcc -fPIC -shared -O2 nlp.c -o nlp.so -L {problem.model_external_shared_lib_dir} -l {problem.model_external_shared_lib_name}", shell=True).wait()
        else:
            subprocess.Popen("gcc -fPIC -shared -O2 nlp.c -o nlp.so", shell=True).wait()
        self.solver = nlpsol("solver", "ipopt", "./nlp.so", opts)

        # Function to get x and u trajectories from w
        self.trajectories = Function('trajectories', [w], [x_solution, u_solution], ['w'], ['x', 'u'])

    def solve(self, x_initial: np.ndarray, params: List[np.ndarray]):
        if self.has_terminal_cost:
            assert len(params) == self.N+1
        else:
            assert len(params) == self.N

        if self.xstar is None:
            x0 = np.tile(x_initial, [self.N+1, 1])
        else:
            x0 = self.xstar
            x0[0, :] = x_initial

        if self.ustar is None:
            u0 = np.zeros([self.N, self.n_u])
        else:
            u0 = self.ustar

        # Formulate the initial guess with correct structure:
        w0 = [x0[0, :].T]
        for k in range(self.N):
            w0.append(u0[k, :].T)
            # State at collocation points
            for j in range(self.polydeg):
                w0.append(x0[k+1, :].T)
            w0.append(x0[k+1, :].T)

        if self.problem.get_terminal_cost_params_fun is not None:
            terminal_cost_input = self.problem.ocp_x_to_terminal_state_fun(x0[-1, :], params[-1]).full().T
            terminal_cost_params = self.problem.get_terminal_cost_params_fun(terminal_cost_input)[0]
            params[-1] = np.r_[params[-1], terminal_cost_params]

        w0 = np.concatenate(w0)
        p = np.concatenate(params)
        self.lbw[:self.n_x] = x0[0, :]
        self.ubw[:self.n_x] = x0[0, :]

        sol = self.solver(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p)

        xstar, ustar = self.trajectories(sol["x"])
        self.xstar, self.ustar = xstar.full().T, ustar.full().T
        sol_x, sol_u = self.xstar.copy(), self.ustar.copy()
        
        # Shift solution to use as next initial guess:
        self.xstar[:-1] = self.xstar[1:]
        self.ustar[:-1] = self.ustar[1:]

        return sol_x, sol_u, sol['f']


