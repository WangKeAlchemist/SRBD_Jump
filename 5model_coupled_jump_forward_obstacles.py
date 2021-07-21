import numpy as np
from casadi import *
import utils.transformations as tf
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from utils.polynomial import *
from mpl_toolkits import mplot3d
from scipy.interpolate import interp1d

# TO DO
# 1, set initial for states for warm start, like x and f
# 2, add more constrains
class SingleRigidBodyModel:
    def __init__(self, mass=13.8, size=(0.25, 0.4, 0.25), g=(0, 0, -9.81)):
        self.mass = mass
        self.size = size
        self.g = np.array(g)
        self.I_body = self.local_inertia_tensor(mass, size)
        self.I_body_inv = np.linalg.inv(self.I_body)
        self.foot_width = 0.1
        self.foot_length = 0.16
        self.l_min = 0.3### Swing phase in the air for trajectory
        self.l_max = 0.8

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
        Ld = np.array([0.0, 0.0, 0.0])
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

    # print(np.rad2deg(tf.euler_from_quaternion(np.array(q).flatten())))


class MotionPlanner:
    def __init__(self, model, T=1, N=50, nlp_solver='ipopt'):
        self.model = model

        # define parameters
        self.T = T
        # self.N = N
        self.N1 = 20
        self.N2 = 20
        self.N3 = 20
        self.N_array = np.array([self.N1, self.N2, self.N3])
        # dt = 0.05
        self.N = self.N1 + self.N2 + self.N3
        self.NX = self.N + 1
        self.NU = self.N
        # self.dt = self.T/self.N
        # side jump: 1.05
        self.r_init = np.array([0.0, 0.0, 0.58])
        self.r_final = np.array([0.0, 0.0, 0.58])
        self.p_left_init = np.array([0.0, 0.21, 0.0])
        self.p_right_init = np.array([0.0, -0.21, 0.0])
        self.p_left_final = np.array([0.25, 0.21, 0.0])
        self.p_right_final = np.array([0.25, -0.21, 0.0])
        self.u_s = 1.0 # friction coefficient

        # Optimization solver
        self.opti = Opti()
        self.opti.solver(nlp_solver)

        # Optimization variables
        # states
        self.r = self.opti.variable(3, self.NX)  # body position
        self.q = self.opti.variable(4, self.NX)  # body quaternion
        self.H = self.opti.variable(3, self.NX)  # body linear momentum
        self.L = self.opti.variable(3, self.NX)  # body angular momentum

        self.rd = self.opti.variable(3, self.NX)  # body position
        self.qd = self.opti.variable(4, self.NX)  # body quaternion
        self.Hd = self.opti.variable(3, self.NX)  # body linear momentum
        self.Ld = self.opti.variable(3, self.NX)  # body angular momentum

        # inputs
        self.dt = self.opti.variable(1, 3)
        p_left = self.opti.variable(3, self.NU)
        p_right = self.opti.variable(3, self.NU)
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

        p1_guess = np.zeros((3, self.N))
        p2_guess = np.zeros((3, self.N))
        p3_guess = np.zeros((3, self.N))
        p4_guess = np.zeros((3, self.N))
        p5_guess = np.zeros((3, self.N))
        p6_guess = np.zeros((3, self.N))
        p7_guess = np.zeros((3, self.N))
        p8_guess = np.zeros((3, self.N))

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

        for i in range(self.N):
            if i < self.N1:
                p1_guess[:,i] = p1_init
                p2_guess[:,i] = p2_init
                p3_guess[:,i] = p3_init
                p4_guess[:,i] = p4_init
                p5_guess[:,i] = p5_init
                p6_guess[:,i] = p6_init
                p7_guess[:,i] = p7_init
                p8_guess[:,i] = p8_init
            elif self.N1 <= i < self.N1 + self.N2:
                p1_guess[:,i] = p1_init + (p1_final-p1_init)/self.N2*(i+1-self.N1)
                p2_guess[:,i] = p2_init + (p2_final-p2_init)/self.N2*(i+1-self.N1)
                p3_guess[:,i] = p3_init + (p3_final-p3_init)/self.N2*(i+1-self.N1)
                p4_guess[:,i] = p4_init + (p4_final-p4_init)/self.N2*(i+1-self.N1)
                p5_guess[:,i] = p5_init + (p5_final-p5_init)/self.N2*(i+1-self.N1)
                p6_guess[:,i] = p6_init + (p6_final-p6_init)/self.N2*(i+1-self.N1)
                p7_guess[:,i] = p7_init + (p7_final-p7_init)/self.N2*(i+1-self.N1)
                p8_guess[:,i] = p8_init + (p8_final-p8_init)/self.N2*(i+1-self.N1)
            else:
                p1_guess[:,i] = p1_final
                p2_guess[:,i] = p2_final
                p3_guess[:,i] = p3_final
                p4_guess[:,i] = p4_final
                p5_guess[:,i] = p5_final
                p6_guess[:,i] = p6_final
                p7_guess[:,i] = p7_final
                p8_guess[:,i] = p8_final

        # Optimization costs
        # costs = sumsqr(diff(f1)) + sumsqr(diff(f2)) + sumsqr(diff(f3)) + sumsqr(diff(f4))
        # costs = sumsqr(f) #+ sumsqr(p)
        # unit_quaternion_cost = 0
        # for k in range(self.N):
        #     unit_quaternion_cost += sumsqr(q[:, k])-1
        # costs = sumsqr(f)
        # costs = sumsqr(diff(f1)) + sumsqr(diff(f2)) + sumsqr(diff(f3)) + sumsqr(diff(f4))
        # costs = sumsqr(diff(self.H.T)) + sumsqr(diff(self.L.T)) + 5*sumsqr(self.r) + sumsqr(diff(f1.T)) + sumsqr(diff(f2.T)) + sumsqr(self.rd) \
        #         + sumsqr(diff(f3.T)) + sumsqr(diff(f4.T)) + sumsqr(diff(f5.T)) + sumsqr(diff(f6.T)) + sumsqr(diff(f7.T)) + sumsqr(diff(f8.T)) \
        #         + sumsqr(self.dt) + sumsqr(diff(p_left.T)) + sumsqr(diff(p_right.T))
        costs = sumsqr(diff(self.H.T)) + sumsqr(diff(self.L.T)) + 5*sumsqr(self.r) + sumsqr(diff(f1.T)) + sumsqr(diff(f2.T)) + sumsqr(self.rd) \
                + sumsqr(diff(f3.T)) + sumsqr(diff(f4.T)) + sumsqr(diff(f5.T)) + sumsqr(diff(f6.T)) + sumsqr(diff(f7.T)) + sumsqr(diff(f8.T)) \
                + sumsqr(diff(p_left.T)) + sumsqr(diff(p_right.T))# + sumsqr(self.r[2,self.N1:(self.N1+self.N2)]-p_left[2,self.N1:(self.N1+self.N2)]) + sumsqr(self.r[2,:-1]-p_right[2,:])
        self.opti.minimize(costs)

        # dynamic constraints
        for i in range(3):
            for j in range(self.N_array[i]):
                if i ==0:
                    k = j
                elif i == 1:
                    k = j + self.N_array[i-1]
                elif i == 2:
                    k = j + self.N_array[i-1]+ self.N_array[i-2]
                self.rd[:,k], self.qd[:,k], self.Hd[:,k], self.Ld[:,k] = self.model.dynamics(self.r[:, k], self.q[:, k], self.H[:, k], self.L[:, k],
                                                     f1[:, k], f2[:, k], f3[:, k], f4[:, k], f5[:, k], f6[:, k],
                                                     f7[:, k], f8[:, k], p1_guess[:,k], p2_guess[:,k], p3_guess[:,k], p4_guess[:,k], p5_guess[:,k], p6_guess[:,k],
                                                     p7_guess[:,k], p8_guess[:,k])
                self.opti.subject_to(self.r[:, k + 1] == self.r[:, k] + self.rd[:,k] * self.dt[i])
                self.opti.subject_to(self.q[:, k + 1] == self.q[:, k] + self.qd[:,k] * self.dt[i])
                self.opti.subject_to(self.H[:, k + 1] == self.H[:, k] + self.Hd[:,k] * self.dt[i])
                self.opti.subject_to(self.L[:, k + 1] == self.L[:, k] + self.Ld[:,k] * self.dt[i])

        # unit quaternion constraints
        for k in range(self.N):
            self.opti.subject_to(sumsqr(self.q[:, k]) == 1.0)
            # self.opti.subject_to(sumsqr(self.q[:, k]) <= 1.01)
        for j in range(3):
            self.opti.subject_to(self.dt[j] >= 0.015)
            self.opti.subject_to(self.dt[j] <= 0.15)

        # constraints on contact positions
        for k in range(self.N):
            # self.opti.subject_to(-0.1 <= self.r[1,k])
            # self.opti.subject_to(self.r[1,k] <= 0.1)
            # self.opti.subject_to(p_left[1,k] == 0.15)
            # self.opti.subject_to(p_right[1,k] == -0.15)
            self.opti.subject_to(p_left[:,self.N1-1] == p_left[:,self.N1])
            self.opti.subject_to(p_right[:,self.N1-1] == p_right[:,self.N1])
            self.opti.subject_to(p_left[:,self.N1+self.N2-1] == p_left[:,self.N1+self.N2])
            self.opti.subject_to(p_right[:,self.N1+self.N2-1] == p_right[:,self.N1+self.N2])

            # self.opti.subject_to(self.r[2, k] > 0.1)
            if k < self.N1:
                self.opti.subject_to(p_left[:,k] == self.p_left_init)
                self.opti.subject_to(p_right[:,k] == self.p_right_init)
                # leg length constraint
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_init) >= self.model.l_min ** 2)
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_init) <= self.model.l_max ** 2)
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_init) >= self.model.l_min ** 2)
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_init) <= self.model.l_max ** 2)
                # friction constraint, no slip
                # self.opti.subject_to(self.u_s*f1[2,k]**2 - (f1[0,k]**2+f1[1,k]**2)>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 - f1[0,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 + f1[0,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 - f1[1,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 + f1[1,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 - f2[0,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 + f2[0,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 - f2[1,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 + f2[1,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 - f3[0,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 + f3[0,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 - f3[1,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 + f3[1,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 - f4[0,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 + f4[0,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 - f4[1,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 + f4[1,k]>=0)
                # self.opti.subject_to(self.u_s*f5[2,k]*0.7 - f5[0,k]>=0)
                # self.opti.subject_to(self.u_s*f5[2,k]*0.7 + f5[0,k]>=0)
                # self.opti.subject_to(self.u_s*f5[2,k]*0.7 - f5[1,k]>=0)
                # self.opti.subject_to(self.u_s*f5[2,k]*0.7 + f5[1,k]>=0)
                # self.opti.subject_to(self.u_s*f6[2,k]*0.7 - f6[0,k]>=0)
                # self.opti.subject_to(self.u_s*f6[2,k]*0.7 + f6[0,k]>=0)
                # self.opti.subject_to(self.u_s*f6[2,k]*0.7 - f6[1,k]>=0)
                # self.opti.subject_to(self.u_s*f6[2,k]*0.7 + f6[1,k]>=0)
                # self.opti.subject_to(self.u_s*f7[2,k]*0.7 - f7[0,k]>=0)
                # self.opti.subject_to(self.u_s*f7[2,k]*0.7 + f7[0,k]>=0)
                # self.opti.subject_to(self.u_s*f7[2,k]*0.7 - f7[1,k]>=0)
                # self.opti.subject_to(self.u_s*f7[2,k]*0.7 + f7[1,k]>=0)
                # self.opti.subject_to(self.u_s*f8[2,k]*0.7 - f8[0,k]>=0)
                # self.opti.subject_to(self.u_s*f8[2,k]*0.7 + f8[0,k]>=0)
                # self.opti.subject_to(self.u_s*f8[2,k]*0.7 - f8[1,k]>=0)
                # self.opti.subject_to(self.u_s*f8[2,k]*0.7 + f8[1,k]>=0)
                # unilateral constraint
                self.opti.subject_to(f1[2, k] >= 0)
                self.opti.subject_to(f2[2, k] >= 0)
                self.opti.subject_to(f3[2, k] >= 0)
                self.opti.subject_to(f4[2, k] >= 0)
                self.opti.subject_to(f5[2, k] >= 0)
                self.opti.subject_to(f6[2, k] >= 0)
                self.opti.subject_to(f7[2, k] >= 0)
                self.opti.subject_to(f8[2, k] >= 0)
                # single leg support
                self.opti.subject_to(0.35 < self.r[2,k] - self.p_left_init[2])
                self.opti.subject_to(self.r[2,k] - self.p_left_init[2] < 0.75)
                self.opti.subject_to(0.35 < self.r[2,k] - self.p_right_init[2])
                self.opti.subject_to(self.r[2,k] - self.p_right_init[2] < 0.75)

            elif self.N1 <= k <= self.N1+self.N2-1:
                self.opti.subject_to(f1[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f2[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f3[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f4[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f5[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f6[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f7[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f8[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(0.35 < self.r[2,k] - p_left[2, k])
                self.opti.subject_to(self.r[2,k] - p_left[2, k] < 0.75)
                self.opti.subject_to(0.35 < self.r[2,k] - p_right[2, k])
                self.opti.subject_to(self.r[2,k] - p_right[2, k] < 0.75)
                # self.opti.subject_to((p_right[1,k] - p_left[1,k])**2 > 0.25**2)
            elif k > self.N1+self.N2-1:
                self.opti.subject_to(p_left[:,k] == self.p_left_final)
                self.opti.subject_to(p_right[:,k] == self.p_right_final)
                # leg length constraint
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_final) >= self.model.l_min)
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_final) <= self.model.l_max)
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_final) >= self.model.l_min)
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_final) <= self.model.l_max)
                # unilateral constraint
                self.opti.subject_to(f1[2, k] >= 0)
                self.opti.subject_to(f2[2, k] >= 0)
                self.opti.subject_to(f3[2, k] >= 0)
                self.opti.subject_to(f4[2, k] >= 0)
                self.opti.subject_to(f5[2, k] >= 0)
                self.opti.subject_to(f6[2, k] >= 0)
                self.opti.subject_to(f7[2, k] >= 0)
                self.opti.subject_to(f8[2, k] >= 0)
                self.opti.subject_to(0.35 < self.r[2,k] - self.p_left_final[2])
                self.opti.subject_to(self.r[2,k] - self.p_left_final[2] < 0.75)
                self.opti.subject_to(0.35 < self.r[2,k] - self.p_right_final[2])
                self.opti.subject_to(self.r[2,k] - self.p_right_final[2] < 0.75)
                # friction constraint, no slip
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 - f1[0,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 + f1[0,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 - f1[1,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 + f1[1,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 - f2[0,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 + f2[0,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 - f2[1,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 + f2[1,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 - f3[0,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 + f3[0,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 - f3[1,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 + f3[1,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 - f4[0,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 + f4[0,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 - f4[1,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 + f4[1,k]>=0)
                # self.opti.subject_to(self.u_s*f5[2,k]*0.7 - f5[0,k]>=0)
                # self.opti.subject_to(self.u_s*f5[2,k]*0.7 + f5[0,k]>=0)
                # self.opti.subject_to(self.u_s*f5[2,k]*0.7 - f5[1,k]>=0)
                # self.opti.subject_to(self.u_s*f5[2,k]*0.7 + f5[1,k]>=0)
                # self.opti.subject_to(self.u_s*f6[2,k]*0.7 - f6[0,k]>=0)
                # self.opti.subject_to(self.u_s*f6[2,k]*0.7 + f6[0,k]>=0)
                # self.opti.subject_to(self.u_s*f6[2,k]*0.7 - f6[1,k]>=0)
                # self.opti.subject_to(self.u_s*f6[2,k]*0.7 + f6[1,k]>=0)
                # self.opti.subject_to(self.u_s*f7[2,k]*0.7 - f7[0,k]>=0)
                # self.opti.subject_to(self.u_s*f7[2,k]*0.7 + f7[0,k]>=0)
                # self.opti.subject_to(self.u_s*f7[2,k]*0.7 - f7[1,k]>=0)
                # self.opti.subject_to(self.u_s*f7[2,k]*0.7 + f7[1,k]>=0)
                # self.opti.subject_to(self.u_s*f8[2,k]*0.7 - f8[0,k]>=0)
                # self.opti.subject_to(self.u_s*f8[2,k]*0.7 + f8[0,k]>=0)
                # self.opti.subject_to(self.u_s*f8[2,k]*0.7 - f8[1,k]>=0)
                # self.opti.subject_to(self.u_s*f8[2,k]*0.7 + f8[1,k]>=0)

        # boundary constraints
        self.opti.subject_to(self.r[:, 0] == self.r_init)
        self.opti.subject_to(self.q[:, 0] == np.array([0, 0, 0, 1]))
        self.opti.subject_to(self.H[:, 0] == np.array([0, 0, 0]))
        self.opti.subject_to(self.L[:, 0] == np.array([0, 0, 0]))

        self.opti.subject_to(self.r[:, -1] == self.r_final)
        # self.opti.subject_to(self.q[:, -1] == tf.quaternion_from_euler(0.0,0.0,1.57))
        self.opti.subject_to(self.H[:, -1] == np.array([0,0,0.0]))
        # self.opti.subject_to(L[:, -1] == np.array([0,0.1,0]))

        self.set_initial_solution()
        f_i = np.zeros((3, self.NU))
        self.opti.set_initial(f1, f_i)
        self.opti.set_initial(f2, f_i)
        self.opti.set_initial(f3, f_i)
        self.opti.set_initial(f4, f_i)
        self.opti.set_initial(f5, f_i)
        self.opti.set_initial(f6, f_i)
        self.opti.set_initial(f7, f_i)
        self.opti.set_initial(f8, f_i)
        p_left_initialize = np.zeros((3, self.NU))
        p_right_initialize = np.zeros((3, self.NU))
        for i in range(self.NU):
            if k < self.N1:
                p_left_initialize[:,k] = self.p_left_init
                p_right_initialize[:,k] = self.p_right_init
            elif self.N1 <= k <= self.N1+self.N2-1:
                p_left_initialize[:,k] = self.p_left_init + (k-self.N1)/self.N2*(self.p_left_final-self.p_left_init)
                p_right_initialize[:,k] = self.p_right_init + (k-self.N1)/self.N2*(self.p_right_final-self.p_right_init)
            else:
                p_left_initialize[:,k] = self.p_left_final
                p_right_initialize[:,k] = self.p_right_final
        self.opti.set_initial(p_left, p_left_initialize)
        self.opti.set_initial(p_right, p_right_initialize)

        options = {"ipopt.max_iter": 20000, "ipopt.warm_start_init_point": "yes"}
        # "verbose": True, "ipopt.print_level": 0, "print_out": False, "print_in": False, "print_time": False,"ipopt.hessian_approximation": "limited-memory",
        self.opti.solver("ipopt", options)

        solution = self.opti.solve()

        # plot
        self.plot(solution.value(self.r),
                  solution.value(self.q),
                  solution.value(self.H),
                  solution.value(self.L),
                  solution.value(self.dt),
                  solution.value(self.rd),
                  solution.value(self.qd),
                  solution.value(self.Hd),
                  solution.value(self.Ld),
                  solution.value(f1),
                  solution.value(f2),
                  solution.value(f3),
                  solution.value(f4),
                  solution.value(f5),
                  solution.value(f6),
                  solution.value(f7),
                  solution.value(f8),
                  solution.value(p_left),
                  solution.value(p_right))

        self.trajectory_generation(solution.value(self.r),solution.value(self.rd),solution.value(p_left), solution.value(p_right), solution.value(self.dt), solution.value(self.Hd))

    def set_initial_solution(self):
        # dT_i = np.ones((1, 3)) * 0.025
        r_i = np.zeros((3, self.NX))
        for i in range(self.NX):
                r_i[:,i] = self.r_init + i/self.NX*(self.r_final - self.r_init)
        # self.opti.set_initial(self.U, U_i)
        # self.opti.set_initial(self.dt, dT_i)
        self.opti.set_initial(self.r, r_i)

    def trajectory_generation(self, r, rd, p_left, p_right, dt, Hd):
        dt_plot = np.ones(self.NU)
        dt_plot = np.concatenate(([0.0], dt_plot))
        dt_plot[1:self.N1] *= dt[0]
        dt_plot[self.N1:self.N1+self.N2] *= dt[1]
        dt_plot[self.N1+self.N2:] *= dt[2]
        for i in range(1, self.NX):
            dt_plot[i] += dt_plot[i - 1]
        dt_plot_control = dt_plot[1:]
        # p_left = np.concatenate((p_left, p_left[:,-1]), axis=1)
        # p_right = np.concatenate((p_right, p_right[:,-1]), axis=1)
        # p_left = np.hstack((p_left[:,:], p_left[:,-1]))
        p_left = np.concatenate((p_left, np.reshape(self.p_left_final, (3,1))), axis=1)
        p_right = np.concatenate((p_right, np.reshape(self.p_right_final, (3,1))), axis=1)
        phase = np.ones(self.NX)
        # double support phase(1) / right support phase(1)
        phase[:self.N1] *= 1.0
        # flight phase
        phase[self.N1:self.N1+self.N2] *= 3.0
        # double support phase
        phase[self.N1+self.N2:] *= 1.0
        # np.concatenate((p_left, np.reshape(self.p_left_final, (3,1))), axis=1)
        sample_num = int(dt_plot[-1] / 0.001)
        comx = interp1d(dt_plot, r[0,:], kind='cubic')
        comy = interp1d(dt_plot, r[1,:], kind='cubic')
        comz = interp1d(dt_plot, r[2,:], kind='cubic')
        comdx = interp1d(dt_plot, rd[0,:], kind='quadratic')
        comdy = interp1d(dt_plot, rd[1,:], kind='quadratic')
        comdz = interp1d(dt_plot, rd[2,:], kind='quadratic')
        comddx = interp1d(dt_plot, Hd[0,:]/self.model.mass, kind='linear')
        comddy = interp1d(dt_plot, Hd[1,:]/self.model.mass, kind='linear')
        comddz = interp1d(dt_plot, Hd[2,:]/self.model.mass, kind='linear')
        lfootx = interp1d(dt_plot, p_left[0,:], kind='linear')
        lfooty = interp1d(dt_plot, p_left[1,:], kind='linear')
        lfootz = interp1d(dt_plot, p_left[2,:], kind='linear')
        rfootx = interp1d(dt_plot, p_right[0,:], kind='linear')
        rfooty = interp1d(dt_plot, p_right[1,:], kind='linear')
        rfootz = interp1d(dt_plot, p_right[2,:], kind='linear')
        phase_t = interp1d(dt_plot, phase, kind='linear')
        sample_t = np.linspace(0, dt_plot[-1], num=sample_num, endpoint=True)
        sample_comx = comx(sample_t)
        sample_comy = comy(sample_t)
        sample_comz = comz(sample_t)
        sample_comdx = comdx(sample_t)
        sample_comdy = comdy(sample_t)
        sample_comdz = comdz(sample_t)
        sample_comddx = comddx(sample_t)
        sample_comddy = comddy(sample_t)
        sample_comddz = comddz(sample_t)
        sample_lfootx = lfootx(sample_t)
        sample_lfooty = lfooty(sample_t)
        sample_lfootz = lfootz(sample_t)
        sample_rfootx = rfootx(sample_t)
        sample_rfooty = rfooty(sample_t)
        sample_rfootz = rfootz(sample_t)
        sample_phase = phase_t(sample_t)
        sample_COM = np.vstack((sample_comx, sample_comy, sample_comz))
        sample_dCOM = np.vstack((sample_comdx, sample_comdy, sample_comdz))
        sample_ddCOM = np.vstack((sample_comddx, sample_comddy, sample_comddz))
        sample_lFOOT = np.vstack((sample_lfootx, sample_lfooty, sample_lfootz))
        sample_rFOOT = np.vstack((sample_rfootx, sample_rfooty, sample_rfootz))
        for i, v in enumerate(sample_phase):
            if v <= 2.0:
                sample_phase[i] = 0
            else:
                # double support is 0, flight phase is 1, right support is 1, left support is -1
                sample_phase[i] = 3
                # elif 2.0 < v < 3.0:
                #     sample_phase[i] = 0
        fig, axs = plt.subplots(2, 2)
        # axs = fig.gca(projection='3d')
        axs[0][0].plot(sample_COM.T)
        axs[0][0].plot(sample_phase.T)
        axs[0][0].legend(['rx', 'ry', 'rz'])
        axs[1][0].plot(sample_lFOOT.T)
        axs[1][0].legend(['px', 'py', 'pz'])
        axs[0][1].plot(sample_rFOOT.T)
        axs[0][1].legend(['px', 'py', 'pz'])
        axs[1][1].plot(dt_plot)
        axs[1][1].legend(['dt'])
        # fig= plt.plot(sample_dCOM.T)
        # axs[0][0].plot(sample_dCOM.T)
        # fig.legend(['drx', 'dry', 'drz'])
        np.savez('data_forward_up_30_higher.npz', copl=sample_lFOOT, copr=sample_rFOOT, time=sample_t, com=sample_COM, dcom=sample_dCOM, ddcom=sample_ddCOM, phase=sample_phase)
        plt.show()

    def plot(self, r, q, H, L, dt, rd, qd, Hd, Ld, f1, f2, f3, f4, f5, f6, f7, f8, p1, p2):#, p1, p2, p3, p4, p5, p6, p7, p8):
        dt_plot = np.ones(self.NU)
        dt_plot = np.concatenate(([0.0], dt_plot))
        dt_plot[1:self.N1] *= dt[0]
        dt_plot[self.N1:self.N1+self.N2] *= dt[1]
        dt_plot[self.N1+self.N2:] *= dt[2]
        for i in range(1, self.NX):
            dt_plot[i] += dt_plot[i - 1]

        dt_plot_control = dt_plot[1:]
        fig, axs = plt.subplots(3, 2)
        # axs = fig.gca(projection='3d')
        axs[0][0].plot(dt_plot, r.T)
        axs[0][0].legend(['rx', 'ry', 'rz'])

        axs[0][1].plot(dt_plot, q.T)
        axs[0][1].plot(dt_plot, np.linalg.norm(q, axis=0))
        axs[0][1].legend(['qx', 'qy', 'qz', 'qw', 'norm'])

        axs[1][0].plot(dt_plot, H.T)
        axs[1][0].legend(['Hx', 'Hy', 'Hz'])

        axs[1][1].plot(dt_plot, L.T)
        axs[1][1].legend(['Lx', 'Ly', 'Lz'])

        axs[2][0].plot(dt_plot)
        axs[2][0].legend(['dt'])

        fig, axs = plt.subplots(2, 2)
        # axs = fig.gca(projection='3d')
        axs[0][0].plot(dt_plot, rd.T)
        axs[0][0].legend(['drx', 'dry', 'drz'])

        axs[0][1].plot(dt_plot, qd.T)
        axs[0][1].plot(np.linalg.norm(qd, axis=0))
        axs[0][1].legend(['dqx', 'dqy', 'dqz', 'dqw', 'norm'])

        axs[1][0].plot(dt_plot, Hd.T)
        axs[1][0].legend(['dHx', 'dHy', 'dHz'])

        axs[1][1].plot(dt_plot, Ld.T)
        axs[1][1].legend(['dLx', 'dLy', 'dLz'])

        fig, axs = plt.subplots(4, 2)
        axs[0][0].plot(dt_plot_control, f1.T)
        axs[0][0].legend(['f1x', 'f1y', 'f1z'])

        axs[1][0].plot(dt_plot_control, f2.T)
        axs[1][0].legend(['f2x', 'f2y', 'f2z'])

        axs[2][0].plot(dt_plot_control, f3.T)
        axs[2][0].legend(['f3x', 'f3y', 'f3z'])

        axs[3][0].plot(dt_plot_control, f4.T)
        axs[3][0].legend(['f4x', 'f4y', 'f4z'])

        axs[0][1].plot(dt_plot_control, f5.T)
        axs[0][1].legend(['f5x', 'f5y', 'f5z'])

        axs[1][1].plot(dt_plot_control, f6.T)
        axs[1][1].legend(['f6x', 'f6y', 'f6z'])

        axs[2][1].plot(dt_plot_control, f7.T)
        axs[2][1].legend(['f7x', 'f7y', 'f7z'])

        axs[3][1].plot(dt_plot_control, f8.T)
        axs[3][1].legend(['f8x', 'f8y', 'f8z'])

        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(dt_plot_control, p1.T)
        axs[0][0].legend(['p1x', 'p1y', 'p1z'])
        #
        axs[1][0].plot(dt_plot_control, p2.T)
        axs[1][0].legend(['p2x', 'p2y', 'p2z'])

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.set_aspect('equal')
        ax.set_xlim3d([-0.5, 0.5])
        ax.set_ylim3d([-0.5, 0.5])
        ax.set_zlim3d([-0.05, 1.0])
        ax.plot3D(r[0,:].T, r[1,:].T, r[2,:].T, 'gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d([-0.5, 0.5])
        ax.set_ylim3d([-0.5, 0.5])
        ax.set_zlim3d([-0.05, 1.0])
        # ax.set_aspect('equal')
        ax.plot3D(p1[0,:].T, p1[1,:].T, p1[2,:].T, 'gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #
        # axs[2][1].plot(p3.T)
        # axs[2][1].legend(['p3x', 'p3y', 'p3z'])
        #
        # axs[3][1].plot(p4.T)
        # axs[3][1].legend(['p4x', 'p4y', 'p4z'])

        # fig, axs = plt.subplots(4, 2)
        # axs[0][0].plot(f5.T)
        # axs[0][0].legend(['f5x', 'f5y', 'f5z'])
        #
        # axs[1][0].plot(f6.T)
        # axs[1][0].legend(['f6x', 'f6y', 'f6z'])
        #
        # axs[2][0].plot(f7.T)
        # axs[2][0].legend(['f7x', 'f7y', 'f7z'])
        #
        # axs[3][0].plot(f8.T)
        # axs[3][0].legend(['f8x', 'f8y', 'f8z'])

        # axs[0][1].plot(p5.T)
        # axs[0][1].legend(['p5x', 'p5y', 'p5z'])
        #
        # axs[1][1].plot(p6.T)
        # axs[1][1].legend(['p6x', 'p6y', 'p6z'])
        #
        # axs[2][1].plot(p7.T)
        # axs[2][1].legend(['p7x', 'p7y', 'p7z'])
        #
        # axs[3][1].plot(p8.T)
        # axs[3][1].legend(['p8x', 'p8y', 'p8z'])

        plt.show()

if __name__ == "__main__":
    single_rigid_body_model = SingleRigidBodyModel()
    motion_planner = MotionPlanner(model=single_rigid_body_model)
