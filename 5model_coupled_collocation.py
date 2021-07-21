import numpy as np
import numpy.matlib
import time
from casadi import *
import utils.transformations as tf
import matplotlib.pyplot as plt

class SingleRigidBodyModel:
    def __init__(self, mass=13.8, size=(0.3, 0.3, 0.3), g=(0, 0, -9.81)):
        self.mass = mass
        self.size = size
        self.g = np.array(g)
        self.I_body = self.local_inertia_tensor(mass, size)
        self.I_body_inv = np.linalg.inv(self.I_body)
        self.foot_width = 0.10
        self.foot_length = 0.165
        self.l_min = 0.3
        self.l_max = 0.7

    def simple_dynamics(self, r, q, H, L, f, p):
        rd = H / self.mass
        # q = q/norm_2(q)
        # q_unit = q/np.linalg.norm(q)
        R = self.quaternion_to_rotation_matrix(q)
        I = R @ self.I_body @ R.T
        I_inv = R @ self.I_body_inv @ R.T
        omega = I_inv @ L
        qd = 1 / 2 * self.Q(q) @ omega
        Hd = f + self.mass * self.g
        Ld = cross(f, r - p)
        return rd, qd, Hd, Ld

    # dynamics for standing
    def dynamics(self, r, q, H, L, f1, f2, f3, f4, f5, f6, f7, f8, p1, p2, p3, p4, p5, p6, p7, p8):
        rd = H / self.mass
        # q = q/norm_2(q)
        # q_unit = q/np.linalg.norm(q)
        R = self.quaternion_to_rotation_matrix(q)
        I = R @ self.I_body @ R.T
        I_inv = R @ self.I_body_inv @ R.T
        omega = I_inv @ L
        qd = 1 / 2 * self.Q(q) @ omega
        Hd = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + self.mass * self.g
        Ld = cross(f1, r - p1) + cross(f2, r - p2) + cross(f3, r - p3) + cross(f4, r - p4) + cross(f5, r - p5) + cross(
            f6, r - p6) + cross(f7, r - p7) + cross(f8, r - p8)
        # Ld = np.cross(f1, r - p1) + np.cross(f2, r - p2) + np.cross(f3, r - p3) + np.cross(f4, r - p4)
        return rd, qd, Hd, Ld

    # dynamics for standing
    def dynamics_inAir(self, r, q, H, L):
        rd = H / self.mass
        # q = q/norm_2(q)
        # q_unit = q/np.linalg.norm(q)
        R = self.quaternion_to_rotation_matrix(q)
        I = R @ self.I_body @ R.T
        I_inv = R @ self.I_body_inv @ R.T
        omega = I_inv @ L
        qd = 1 / 2 * self.Q(q) @ omega
        Hd = self.mass * self.g
        Ld = np.zeros(3)
        return rd, qd, Hd, Ld

    def Q(self, quaternion):
        qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        Q = blockcat([[qw, -qz, qy],
                      [qz, qw, -qx],
                      [-qy, qx, qw],
                      [-qx, -qy, -qz]])
        return Q

    def quaternion_derivative(self, quaternion, angular_velocity):
        q_dot = 1 / 2 * self.Q(quaternion) @ angular_velocity
        return q_dot

    def quaternion_to_rotation_matrix(self, q):
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        # R = np.array([[1 - 2 * (qy ** 2 + qz ** 2), 2 * qx * qy - 2 * qw * qz, 2 * qw * qy + 2 * qx * qz],
        #               [2 * qx * qy + 2 * qw * qz, 1 - 2 * (qx ** 2 + qz ** 2), 2 * qy * qz - 2 * qw * qx],
        #               [2 * qx * qz - 2 * qw * qy, 2 * qw * qx + 2 * qy * qz, 1 - 2 * (qx ** 2 + qy ** 2)]])
        R = blockcat([[1 - 2 * (qy ** 2 + qz ** 2), 2 * qx * qy - 2 * qw * qz, 2 * qw * qy + 2 * qx * qz],
                      [2 * qx * qy + 2 * qw * qz, 1 - 2 * (qx ** 2 + qz ** 2), 2 * qy * qz - 2 * qw * qx],
                      [2 * qx * qz - 2 * qw * qy, 2 * qw * qx + 2 * qy * qz, 1 - 2 * (qx ** 2 + qy ** 2)]])
        return R

    def local_inertia_tensor(self, mass, size):
        x, y, z = size[0], size[1], size[2]
        Ixx = 1 / 12 * mass * (y ** 2 + z ** 2)
        Iyy = 1 / 12 * mass * (x ** 2 + z ** 2)
        Izz = 1 / 12 * mass * (x ** 2 + y ** 2)
        I = np.diag([Ixx, Iyy, Izz])
        return I

    def print_parameters(self):
        print('mass:', self.mass)
        print('size:', self.size)
        print('I_body:\n', self.I_body)
        print('I_body_inv:\n', self.I_body_inv)

class MotionPlanner:
    def __init__(self, model, T=2, N=50, nlp_solver='ipopt'):
        self.model = model

        # define parameters
        self.T = T
        # self.N = N
        self.N1 = 15
        self.N2 = 5
        self.N3 = 15
        # N = 20  # number of control intervals
        d = 1
        self.N = self.N1 + self.N2 + self.N3
        self.NX = self.N + 1
        self.NU = self.N
        # self.dt = self.T/self.N
        self.r_init = np.array([0.0, 0.0, 0.58])
        self.r_final = np.array([0.1, 0.0, 0.58])
        self.p_left_init = np.array([0.0, 0.21, 0.0])
        self.p_right_init = np.array([0.0, -0.21, 0.0])
        self.p_left_final = np.array([0.1, 0.21, 0.0])
        self.p_right_final = np.array([0.1, -0.21, 0.0])
        self.u_s = 1.0 # friction coefficient

        # Optimization solver
        self.opti = Opti()
        self.opti.solver(nlp_solver)

        tau = collocation_points(d, 'legendre')
        # Collocation linear maps
        [C, D, B] = collocation_coeff(tau)
        # Declare model variables
        r_s = MX.sym('r_s', 3)
        q_s = MX.sym('q_s', 4)
        L_s = MX.sym('L_s', 3)
        H_s = MX.sym('H_s', 3)
        dr_1 = MX.sym('dr_1', 3)
        dq_1 = MX.sym('dq_1', 4)
        dL_1 = MX.sym('dL_1', 3)
        dH_1 = MX.sym('dH_1', 3)

        f1_s = MX.sym('f1_s', 3)
        f2_s = MX.sym('f2_s', 3)
        f3_s = MX.sym('f3_s', 3)
        f4_s = MX.sym('f4_s', 3)
        f5_s = MX.sym('f5_s', 3)
        f6_s = MX.sym('f6_s', 3)
        f7_s = MX.sym('f7_s', 3)
        f8_s = MX.sym('f8_s', 3)
        p1_s = MX.sym('p1_s', 3)
        p2_s = MX.sym('p2_s', 3)
        p3_s = MX.sym('p3_s', 3)
        p4_s = MX.sym('p4_s', 3)
        p5_s = MX.sym('p5_s', 3)
        p6_s = MX.sym('p6_s', 3)
        p7_s = MX.sym('p7_s', 3)
        p8_s = MX.sym('p8_s', 3)

        p1_init = self.p_left_init + np.array([-self.model.foot_length / 2, self.model.foot_width / 2, 0.0])
        p2_init = self.p_left_init + np.array([self.model.foot_length / 2, self.model.foot_width / 2, 0.0])
        p3_init = self.p_left_init + np.array([self.model.foot_length / 2, -self.model.foot_width / 2, 0.0])
        p4_init = self.p_left_init + np.array([-self.model.foot_length / 2, -self.model.foot_width / 2, 0.0])
        p5_init = self.p_right_init + np.array([-self.model.foot_length / 2, self.model.foot_width / 2, 0.0])
        p6_init = self.p_right_init + np.array([self.model.foot_length / 2, self.model.foot_width / 2, 0.0])
        p7_init = self.p_right_init + np.array([self.model.foot_length / 2, -self.model.foot_width / 2, 0.0])
        p8_init = self.p_right_init + np.array([-self.model.foot_length / 2, -self.model.foot_width / 2, 0.0])

        p1_final = self.p_left_final + np.array([-self.model.foot_length / 2, self.model.foot_width / 2, 0.0])
        p2_final = self.p_left_final + np.array([self.model.foot_length / 2, self.model.foot_width / 2, 0.0])
        p3_final = self.p_left_final + np.array([self.model.foot_length / 2, -self.model.foot_width / 2, 0.0])
        p4_final = self.p_left_final + np.array([-self.model.foot_length / 2, -self.model.foot_width / 2, 0.0])
        p5_final = self.p_right_final + np.array([-self.model.foot_length / 2, self.model.foot_width / 2, 0.0])
        p6_final = self.p_right_final + np.array([self.model.foot_length / 2, self.model.foot_width / 2, 0.0])
        p7_final = self.p_right_final + np.array([self.model.foot_length / 2, -self.model.foot_width / 2, 0.0])
        p8_final = self.p_right_final + np.array([-self.model.foot_length / 2, -self.model.foot_width / 2, 0.0])

        dr_1, dq_1, dH_1, dL_1 = self.model.dynamics(r_s,q_s,H_s,L_s,f1_s,f2_s,f3_s,f4_s,f5_s,f6_s,f7_s,f8_s,p1_s,p2_s,p3_s,p4_s,p5_s,p6_s,p7_s,p8_s)
    
        # Model equations
        x_s = vertcat(r_s, q_s, H_s, L_s)
        xdot_1 = vertcat(dr_1, dq_1, dH_1, dL_1)
        # Objective term
        L1 = sumsqr(dr_1) + sumsqr(dL_1) + sumsqr(dH_1)
        # Continuous time dynamics
        f_1 = Function('f', [r_s,q_s,H_s,L_s,f1_s,f2_s,f3_s,f4_s,f5_s,f6_s,f7_s,f8_s,p1_s,p2_s,p3_s,p4_s,p5_s,p6_s,p7_s,p8_s], [xdot_1, L1])

        # Optimization variables
        # states
        self.r = self.opti.variable(3, self.NX)  # body position
        self.q = self.opti.variable(4, self.NX)  # body quaternion
        self.H = self.opti.variable(3, self.NX)  # body linear momentum
        self.L = self.opti.variable(3, self.NX)  # body angular momentum

        # inputs
        self.dt = self.opti.variable(1, self.NU)
        # Left Foot: leftbottom=f1, lefttop=f2, righttop=f3, rightbottom= f4
        # Right Foot: leftbottom=f5, lefttop=f6, righttop=f7, rightbottom= f8
        f1 = self.opti.variable(3, self.NU)
        f2 = self.opti.variable(3, self.NU)
        f3 = self.opti.variable(3, self.NU)
        f4 = self.opti.variable(3, self.NU)
        f5 = self.opti.variable(3, self.NU)
        f6 = self.opti.variable(3, self.NU)
        f7 = self.opti.variable(3, self.NU)
        f8 = self.opti.variable(3, self.NU)

        J = 0
        # Formulate the NLP
        for k in range(0, self.NU):
            # Decision variables for helper states at each collocation point
            Rc = self.opti.variable(3, d)
            self.opti.set_initial(Rc, np.tile(self.r_init, (d,1)).T)
            Qc = self.opti.variable(4, d)
            self.opti.set_initial(Qc, np.tile(np.array([0, 0, 0, 1]), (d,1)).T)
            Hc = self.opti.variable(3, d)
            self.opti.set_initial(Hc, np.tile(np.array([0, 0, 0]),(d,1)).T)
            Lc = self.opti.variable(3, d)
            self.opti.set_initial(Lc, np.tile(np.array([0, 0, 0]),(d,1)).T)

            # Evaluate ODE right-hand-side at all helper states
            ode, quad = f_1(Rc, Qc, Hc, Lc, f1[:, k], f2[:, k], f3[:, k], f4[:, k], f5[:, k], f6[:, k], f7[:, k], f8[:, k], p1_init,p2_init,p3_init,p4_init,p5_init,p6_init,p7_init,p8_init)
            # Add contribution to quadrature function
            J += quad @ B * self.dt[k]

            # Get interpolating points of collocation polynomial
            # Z = [Xk, Xc]
            Z = horzcat(vertcat(self.r[:,k],self.q[:,k], self.H[:,k], self.L[:,k]), vertcat(Rc, Qc, Hc, Lc))
            # Get slope of interpolating polynomial (normalized)
            Pidot = Z @ C
            # Match with ODE right-hand-side
            self.opti.subject_to(Pidot == self.dt[k] * ode)
            # State at end of collocation interval
            Xk_end = Z @ D

            # Continuity constraints
            self.opti.subject_to(Xk_end == vertcat(self.r[:,k+1],self.q[:,k+1], self.H[:,k+1], self.L[:,k+1]))

            self.opti.subject_to(self.dt[k] > 0.025)
            self.opti.subject_to(self.dt[k] < 0.25)
            self.opti.subject_to(0.0 <= self.r[0, k])
            # self.opti.subject_to(-0.25 <= self.r[1,k])
            # self.opti.subject_to(self.r[1,k] <= 0.25)
            if k < self.N1:
                # leg length constraint
                self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_init) >= self.model.l_min ** 2)
                self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_init) <= self.model.l_max ** 2)
                self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_init) >= self.model.l_min ** 2)
                self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_init) <= self.model.l_max ** 2)
                # friction constraint, no slip
                # self.opti.subject_to(self.u_s*f1[2,k]**2 - (f1[0,k]**2+f1[1,k]**2)>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 - f1[0,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 + f1[0,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 - f1[1,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 + f1[1,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]**2 - (f2[0,k]**2+f2[1,k]**2)>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]**2 - (f3[0,k]**2+f3[1,k]**2)>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]**2 - (f4[0,k]**2+f4[1,k]**2)>=0)
                # self.opti.subject_to(self.u_s*f5[2,k]**2 - (f5[0,k]**2+f5[1,k]**2)>=0)
                # self.opti.subject_to(self.u_s*f6[2,k]**2 - (f6[0,k]**2+f6[1,k]**2)>=0)
                # self.opti.subject_to(self.u_s*f7[2,k]**2 - (f7[0,k]**2+f7[1,k]**2)>=0)
                # self.opti.subject_to(self.u_s*f8[2,k]**2 - (f8[0,k]**2+f8[1,k]**2)>=0)
                # unilateral constraint
                self.opti.subject_to(f1[2, k] >= 0)
                self.opti.subject_to(f2[2, k] >= 0)
                self.opti.subject_to(f3[2, k] >= 0)
                self.opti.subject_to(f4[2, k] >= 0)
                self.opti.subject_to(f5[2, k] >= 0)
                self.opti.subject_to(f6[2, k] >= 0)
                self.opti.subject_to(f7[2, k] >= 0)
                self.opti.subject_to(f8[2, k] >= 0)
                self.opti.subject_to(0.4 < self.r[2,k] - self.p_left_init[2])
                self.opti.subject_to(self.r[2,k] - self.p_left_init[2] < 0.75)

            elif self.N1 <= k <= self.N1+self.N2-1:
                self.opti.subject_to(f1[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f2[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f3[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f4[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f5[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f6[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f7[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f8[:, k] == np.array([0.0, 0.0, 0.0]))
                # self.opti.subject_to(0.2 < self.r[2,k] - p_left[2, k])
                # self.opti.subject_to(self.r[2,k] - p_left[2, k] < 0.8)
                # self.opti.subject_to(0.2 < self.r[2,k] - p_right[2, k])
                # self.opti.subject_to(self.r[2,k] - p_right[2, k] < 0.8)
            elif k > self.N1+self.N2-1:
                # self.opti.subject_to(p_left[:,k] == self.p_left_final)
                # self.opti.subject_to(p_right[:,k] == self.p_right_final)
                # leg length constraint
                self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_final) >= self.model.l_min)
                self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_final) <= self.model.l_max)
                self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_final) >= self.model.l_min)
                self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_final) <= self.model.l_max)
                # unilateral constraint
                self.opti.subject_to(f1[2, k] >= 0)
                self.opti.subject_to(f2[2, k] >= 0)
                self.opti.subject_to(f3[2, k] >= 0)
                self.opti.subject_to(f4[2, k] >= 0)
                self.opti.subject_to(f5[2, k] >= 0)
                self.opti.subject_to(f6[2, k] >= 0)
                self.opti.subject_to(f7[2, k] >= 0)
                self.opti.subject_to(f8[2, k] >= 0)
                self.opti.subject_to(0.4 < self.r[2,k] - self.p_left_final[2])
                self.opti.subject_to(self.r[2,k] - self.p_left_final[2] < 0.75)

        self.opti.subject_to(self.r[:, 0] == self.r_init)
        self.opti.subject_to(self.q[:, 0] == np.array([0, 0, 0, 1]))
        self.opti.subject_to(self.H[:, 0] == np.array([0, 0, 0]))
        self.opti.subject_to(self.L[:, 0] == np.array([0, 0, 0]))

        self.opti.subject_to(self.r[:, -1] == self.r_final)
        # self.opti.subject_to(self.q[:, -1] == tf.quaternion_from_euler(0.0,0.0,0.0))
        # self.opti.subject_to(self.H[:, -1] == np.array([0,0,0.0]))
        # self.opti.subject_to(self.L[:, -1] == np.array([0,0.0,0]))
        r_i = np.tile(self.r_init, (self.NX,1)).T
        self.opti.set_initial(self.r, r_i)
        q_i =  np.tile(np.array([0, 0, 0, 1]), (self.NX, 1)).T
        self.opti.set_initial(self.q, q_i)
        dT_i = np.ones((1, self.NU)) * 0.1
        self.opti.set_initial(self.dt, dT_i)

        self.opti.minimize(J)
        options = {"ipopt.max_iter": 20000, "ipopt.warm_start_init_point": "yes"}#, "ipopt.mumps_mem_percent": 200
        self.opti.solver('ipopt', options)
        solution = self.opti.solve()
        # plot

        self.plot(solution.value(self.r),
                  solution.value(self.q),
                  solution.value(self.H),
                  solution.value(self.L),
                  solution.value(self.dt),
                  solution.value(f1),
                  solution.value(f2),
                  solution.value(f3),
                  solution.value(f4),
                  solution.value(f5),
                  solution.value(f6),
                  solution.value(f7),
                  solution.value(f8))
                  # solution.value(p_left),
                  # solution.value(p_right))
        # solution.value(p3),
        # solution.value(p4),
        # solution.value(p5),
        # solution.value(p6),
        # solution.value(p7),
        # solution.value(p8))

    def set_initial_solution(self):
        dT_i = np.ones((1, 3)) * 0.1
        r_i = np.zeros((3, self.NX))
        for i in range(self.NX):
            r_i[:, i] = self.r_init + i / self.NX * (self.r_final - self.r_init)
        # self.opti.set_initial(self.U, U_i)
        self.opti.set_initial(self.dt, dT_i)
        self.opti.set_initial(self.r, r_i)

    def plot(self, r, q, H, L, dt, f1, f2, f3, f4, f5, f6, f7, f8):  # , p1, p2, p3, p4, p5, p6, p7, p8):

        fig, axs = plt.subplots(3, 2)
        # axs = fig.gca(projection='3d')
        axs[0][0].plot(r.T)
        axs[0][0].legend(['rx', 'ry', 'rz'])

        axs[0][1].plot(q.T)
        axs[0][1].plot(np.linalg.norm(q, axis=0))
        axs[0][1].legend(['qx', 'qy', 'qz', 'qw', 'norm'])

        axs[1][0].plot(H.T)
        axs[1][0].legend(['Hx', 'Hy', 'Hz'])

        axs[1][1].plot(L.T)
        axs[1][1].legend(['Lx', 'Ly', 'Lz'])

        dt_plot = np.ones(self.NU)
        dt_plot = np.concatenate(([0.0], dt_plot))
        dt_plot[1:self.N1] *= dt[0]
        dt_plot[self.N1:self.N1 + self.N2] *= dt[1]
        dt_plot[self.N1 + self.N2:] *= dt[2]
        for i in range(1, self.NX):
            dt_plot[i] += dt_plot[i - 1]
        axs[2][0].plot(dt_plot)
        axs[2][0].legend(['dt'])

        # fig, axs = plt.subplots(2, 2)
        # # axs = fig.gca(projection='3d')
        # axs[0][0].plot(rd.T)
        # axs[0][0].legend(['drx', 'dry', 'drz'])
        #
        # axs[0][1].plot(qd.T)
        # axs[0][1].plot(np.linalg.norm(qd, axis=0))
        # axs[0][1].legend(['dqx', 'dqy', 'dqz', 'dqw', 'norm'])
        #
        # axs[1][0].plot(Hd.T)
        # axs[1][0].legend(['dHx', 'dHy', 'dHz'])
        #
        # axs[1][1].plot(Ld.T)
        # axs[1][1].legend(['dLx', 'dLy', 'dLz'])

        fig, axs = plt.subplots(4, 2)
        axs[0][0].plot(f1.T)
        axs[0][0].legend(['f1x', 'f1y', 'f1z'])

        axs[1][0].plot(f2.T)
        axs[1][0].legend(['f2x', 'f2y', 'f2z'])

        axs[2][0].plot(f3.T)
        axs[2][0].legend(['f3x', 'f3y', 'f3z'])

        axs[3][0].plot(f4.T)
        axs[3][0].legend(['f4x', 'f4y', 'f4z'])

        axs[0][1].plot(f5.T)
        axs[0][1].legend(['f5x', 'f5y', 'f5z'])

        axs[1][1].plot(f6.T)
        axs[1][1].legend(['f6x', 'f6y', 'f6z'])

        axs[2][1].plot(f7.T)
        axs[2][1].legend(['f7x', 'f7y', 'f7z'])

        axs[3][1].plot(f8.T)
        axs[3][1].legend(['f8x', 'f8y', 'f8z'])
        plt.show()


        # fig, axs = plt.subplots(2, 2)
        # axs[0][0].plot(p1.T)
        # axs[0][0].legend(['p1x', 'p1y', 'p1z'])
        # #
        # axs[1][0].plot(p2.T)
        # axs[1][0].legend(['p2x', 'p2y', 'p2z'])



# Xs = [Xs{:}];
# Us = [Us{:}];

# opti.minimize(J)
#
# opti.solver('ipopt')
#
# sol = opti.solve()
#
# x_opt = sol.value(Xk)
# u_opt = sol.value(Uk)
if __name__ == "__main__":
    single_rigid_body_model = SingleRigidBodyModel()
    motion_planner = MotionPlanner(model=single_rigid_body_model)