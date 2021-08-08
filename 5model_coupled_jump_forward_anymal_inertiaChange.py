import numpy as np
from casadi import *
import utils.transformations as tf
import matplotlib.pyplot as plt
from utils.transformations import euler_from_quaternion, quaternion_slerp
from mpl_toolkits import mplot3d
from scipy.interpolate import interp1d

def angular_vel_quaterion_derivative(q,qd):
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    qdx, qdy, qdz, qdw = qd[0], qd[1], qd[2], qd[3]
    wx = 2*(qw*qdx - qx*qdw - qy*qdz + qz*qdy)
    wy = 2*(qw*qdy + qx*qdz - qy*qdw - qz*qdx)
    wz = 2*(qw*qdz - qx*qdy + qy*qdx -qz*qdw)
    return blockcat([[wx, wy, wz]])

def angular_vel_quaterion_derivative_numerical(q,qd):
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    qdx, qdy, qdz, qdw = qd[0], qd[1], qd[2], qd[3]
    wx = 2*(qw*qdx - qx*qdw - qy*qdz + qz*qdy)
    wy = 2*(qw*qdy + qx*qdz - qy*qdw - qz*qdx)
    wz = 2*(qw*qdz - qx*qdy + qy*qdx -qz*qdw)
    return np.array([wx, wy, wz])

class SingleRigidBodyModel:
    def __init__(self, mass=35.65, g=(0, 0, -9.81)):
        self.mass = mass
        self.g = np.array(g)
        self.I_body = np.array([[1.14402, -0.00138895, 0.0095396],[-0.00138895, 2.28541, -6.77755e-05],[0.0095396,-6.77755e-05, 2.26563]])
        self.I_body_inv = np.linalg.inv(self.I_body)
        self.l_min = 0.3### Swing phase in the air for trajectory
        self.l_max = 0.8
        self.leg_mass = 2.236
        self.x_default = 0.348
        self.y_default = 0.215
        self.z_default = 0.468
        # self.I_body_constant = self.I_body - 4*np.array([[self.leg_mass*(self.x_default-0.04)**2, 0., 0.],
        #                                                 [0., self.leg_mass*self.y_default**2, 0.],
        #                                                 [0., 0., self.leg_mass*(self.z_default/2)**2]])
        self.I_body_constant = self.I_body - np.array([[0.80496, 0., 0.],
                                                        [0., 0.4134, 0.],
                                                        [0., 0., 0.4897]])



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

    def update_inertia(self, pos_lf, pos_lh, pos_rf, pos_rh):
        self.I_body = blockcat([[(pos_lf[0,0]**2+pos_lh[0,0]**2+pos_rf[0,0]**2+pos_rh[0,0]**2)*self.leg_mass, 0.,0.],
                       [0.,(pos_lf[1,0]**2+pos_lh[1,0]**2+pos_rf[1,0]**2+pos_rh[1,0]**2)*self.leg_mass,0.],
                       [0.,0.,(pos_lf[2,0]**2+pos_lh[2,0]**2+pos_rf[2,0]**2+pos_rh[2,0]**2)*self.leg_mass/4.]]) + self.I_body_constant
        self.I_body_inv = inv(self.I_body)
        # self.I_body_inv = blockcat([[1./((pos_lf[0,0]**2+pos_lh[0,0]**2+pos_rf[0,0]**2+pos_rh[0,0]**2)*self.leg_mass+0.33904), 0.,0.],
        #                [0.,1./((pos_lf[1,0]**2+pos_lh[1,0]**2+pos_rf[1,0]**2+pos_rh[1,0]**2)*self.leg_mass+1.872),0.],
        #                [0.,0.,1./((pos_lf[2,0]**2+pos_lh[2,0]**2+pos_rf[2,0]**2+pos_rh[2,0]**2)*self.leg_mass/4+1.77593)]])


    # dynamics for standing
    def dynamics(self, r, q, H, L, f1, f2, f3, f4, p1, p2, p3, p4):
        rd = H / self.mass
        R = self.quaternion_to_rotation_matrix(q)
        I = R @ self.I_body @ R.T
        I_inv = R @ self.I_body_inv @ R.T
        omega = I_inv @ L
        qd = 1 / 2 * self.Q(q) @ omega
        Hd = f1 + f2 + f3 + f4 + self.mass * self.g
        Ld = cross(f1, r - p1) + cross(f2, r - p2) + cross(f3, r - p3) + cross(f4, r - p4)
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

    def print_parameters(self):
        print('mass:', self.mass)
        # print('size:', self.size)
        print('I_body:\n', self.I_body)
        print('I_body_inv:\n', self.I_body_inv)

    # print(np.rad2deg(tf.euler_from_quaternion(np.array(q).flatten())))


class MotionPlanner:
    def __init__(self, model, T=1, N=50, nlp_solver='ipopt'):
        self.model = model

        # define parameters
        self.T = T
        # jump forward with 0.1m, N1=N2=N3=20, tmin=0.02, tmax=0.15
        # jump hopping: N1=N2=N3=15 , tmin=0.02, tmax=0.15
        self.N1 = 15
        self.N2 = 20
        self.N3 = 20
        self.N_array = np.array([self.N1, self.N2, self.N3])
        self.N = self.N1 + self.N2 + self.N3
        self.NX = self.N + 1
        self.NU = self.N
        # self.dt = self.T/self.N
        # side jump: 1.05
        x_default = 0.348
        y_default = 0.215
        z_default = 0.468
        self.final_offset = np.array([0.0, 0.0, 0.0])
        self.r_init = np.array([0.0, 0.0, z_default])
        self.r_final = np.array([0.0, 0.0, z_default]) + self.final_offset
        self.q_init = np.array([0,0,0,1])
        self.q_final = tf.quaternion_from_euler(0, 0, np.pi/2)
        R = self.quaternion_to_rotation_matrix(self.q_final)
        self.p_left_front_init = np.array([x_default, y_default, 0.0])
        self.p_right_front_init = np.array([x_default, -y_default, 0.0])
        self.p_left_rear_init = np.array([-x_default, y_default, 0.0])
        self.p_right_rear_init = np.array([-x_default, -y_default, 0.0])
        self.p_left_front_final = R@np.array([x_default, y_default, 0.0]) + self.final_offset
        self.p_right_front_final = R@np.array([x_default, -y_default, 0.0]) + self.final_offset
        self.p_left_rear_final = R@np.array([-x_default, y_default, 0.0]) + self.final_offset
        self.p_right_rear_final = R@np.array([-x_default, -y_default, 0.0]) + self.final_offset
        self.u_s = 0.7 # friction coefficient
        # Optimization solver
        self.opti = Opti()
        self.opti.solver(nlp_solver)
        # Optimization variables
        # states
        self.r = self.opti.variable(3, self.NX)  # CoM position
        self.q = self.opti.variable(4, self.NX)  # body quaternion
        self.H = self.opti.variable(3, self.NX)  # body linear momentum
        self.L = self.opti.variable(3, self.NX)  # body angular momentum

        self.rd = self.opti.variable(3, self.NX)  # body linear vel
        self.qd = self.opti.variable(4, self.NX)  # body angular vel
        self.Hd = self.opti.variable(3, self.NX)  # body linear momentum
        self.Ld = self.opti.variable(3, self.NX)  # body angular momentum

        # inputs
        self.dt = self.opti.variable(1, 3)
        p_left_front = self.opti.variable(3, self.NU)
        p_left_rear = self.opti.variable(3, self.NU)
        p_right_front = self.opti.variable(3, self.NU)
        p_right_rear = self.opti.variable(3, self.NU)
        # Left Foot: leftbottom=f1, lefttop=f2, righttop=f3, rightbottom= f4
        # Right Foot: leftbottom=f5, lefttop=f6, righttop=f7, rightbottom= f8
        f1 = self.opti.variable(3, self.NU)
        f2 = self.opti.variable(3, self.NU)
        f3 = self.opti.variable(3, self.NU)
        f4 = self.opti.variable(3, self.NU)

        p1_guess = np.zeros((3, self.N))
        p2_guess = np.zeros((3, self.N))
        p3_guess = np.zeros((3, self.N))
        p4_guess = np.zeros((3, self.N))

        p1_init = self.p_left_front_init
        p2_init = self.p_right_front_init
        p3_init = self.p_left_rear_init
        p4_init = self.p_right_rear_init
        p1_final = self.p_left_front_final
        p2_final = self.p_right_front_final
        p3_final = self.p_left_rear_final
        p4_final = self.p_right_rear_final

        for i in range(self.N):
            if i < self.N1:
                p1_guess[:,i] = p1_init
                p2_guess[:,i] = p2_init
                p3_guess[:,i] = p3_init
                p4_guess[:,i] = p4_init
            elif self.N1 <= i < self.N1 + self.N2:
                p1_guess[:,i] = p1_init + (p1_final-p1_init)/self.N2*(i+1-self.N1)
                p2_guess[:,i] = p2_init + (p2_final-p2_init)/self.N2*(i+1-self.N1)
                p3_guess[:,i] = p3_init + (p3_final-p3_init)/self.N2*(i+1-self.N1)
                p4_guess[:,i] = p4_init + (p4_final-p4_init)/self.N2*(i+1-self.N1)
            else:
                p1_guess[:,i] = p1_final
                p2_guess[:,i] = p2_final
                p3_guess[:,i] = p3_final
                p4_guess[:,i] = p4_final
        # dynamic constraints
        for i in range(3):
            for j in range(self.N_array[i]):
                if i ==0:
                    k = j
                elif i == 1:
                    k = j + self.N_array[i-1]
                elif i == 2:
                    k = j + self.N_array[i-1] + self.N_array[i-2]
                # self.model.update_inertia(p_left_front[:,k], p_left_rear[:,k], p_right_front[:,k], p_right_rear[:,k])
                self.rd[:,k], self.qd[:,k], self.Hd[:,k], self.Ld[:,k] = self.model.dynamics(self.r[:, k], self.q[:, k], self.H[:, k], self.L[:, k],
                                                     f1[:, k], f2[:, k], f3[:, k], f4[:, k], p1_guess[:,k], p2_guess[:,k], p3_guess[:,k], p4_guess[:,k])
                self.opti.subject_to(self.r[:, k + 1] == self.r[:, k] + self.rd[:,k] * self.dt[i])
                self.opti.subject_to(self.q[:, k + 1] == self.q[:, k] + self.qd[:,k] * self.dt[i])
                self.opti.subject_to(self.H[:, k + 1] == self.H[:, k] + self.Hd[:,k] * self.dt[i])
                self.opti.subject_to(self.L[:, k + 1] == self.L[:, k] + self.Ld[:,k] * self.dt[i])

        # unit quaternion constraints
        for k in range(self.N):
            self.opti.subject_to(sumsqr(self.q[:, k]) >= 0.99)
            self.opti.subject_to(sumsqr(self.q[:, k]) <= 1.01)
        for j in range(3):
            self.opti.subject_to(self.dt[j] >= 0.025)
            self.opti.subject_to(self.dt[j] <= 0.15)

        # constraints on contact positions
        for k in range(self.N):
            # continualty in phases
            self.opti.subject_to(p_left_front[:,self.N1-1] == p_left_front[:,self.N1])
            self.opti.subject_to(p_left_rear[:,self.N1-1] == p_left_rear[:,self.N1])
            self.opti.subject_to(p_right_front[:,self.N1-1] == p_right_front[:,self.N1])
            self.opti.subject_to(p_right_rear[:,self.N1-1] == p_right_rear[:,self.N1])
            self.opti.subject_to(p_left_front[:,self.N1+self.N2-1] == p_left_front[:,self.N1+self.N2])
            self.opti.subject_to(p_left_rear[:,self.N1+self.N2-1] == p_left_rear[:,self.N1+self.N2])
            self.opti.subject_to(p_right_front[:,self.N1+self.N2-1] == p_right_front[:,self.N1+self.N2])
            self.opti.subject_to(p_right_rear[:,self.N1+self.N2-1] == p_right_rear[:,self.N1+self.N2])
            # kinematic constraint
            self.opti.subject_to(0.31 <= self.r[2, k] - p_left_front[2, k])
            self.opti.subject_to(0.31 <= self.r[2, k] - p_left_rear[2, k])
            self.opti.subject_to(0.31 <= self.r[2, k] - p_right_front[2, k])
            self.opti.subject_to(0.31 <= self.r[2, k] - p_right_rear[2, k])
            self.opti.subject_to(self.r[2, k] - p_left_front[2, k] <= 0.56)
            self.opti.subject_to(self.r[2, k] - p_left_rear[2, k] <= 0.56)
            self.opti.subject_to(self.r[2, k] - p_right_front[2, k] <= 0.56)
            self.opti.subject_to(self.r[2, k] - p_right_rear[2, k] <= 0.56)
            self.opti.subject_to(p_left_front[2, k] >= 0)
            self.opti.subject_to(p_left_rear[2, k] >= 0)
            self.opti.subject_to(p_right_front[2, k] >= 0)
            self.opti.subject_to(p_right_rear[2, k] >= 0)

            if k < self.N1:
                self.opti.subject_to(p_left_front[:,k] == self.p_left_front_init)
                self.opti.subject_to(p_left_rear[:,k] == self.p_left_rear_init)
                self.opti.subject_to(p_right_front[:,k] == self.p_right_front_init)
                self.opti.subject_to(p_right_rear[:,k] == self.p_right_rear_init)
                # leg length constraint
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_init) >= self.model.l_min ** 2)
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_left_init) <= self.model.l_max ** 2)
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_init) >= self.model.l_min ** 2)
                # self.opti.subject_to(sumsqr(self.r[:,k]-self.p_right_init) <= self.model.l_max ** 2)
                # friction constraint, no slip
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 - f1[0,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 + f1[0,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 - f1[1,k]>=0)
                # self.opti.subject_to(self.u_s*f1[2,k]*0.7 + f1[1,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 - f2[0,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 + f2[0,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 - f2[1,k]>=0)
                # self.opti.subject_to(self.u_s*f2[2,k]*0.7 + f2[1,k]>=0)
                # self.opti.subject_to(quaternionself.u_s*f3[2,k]*0.7 - f3[0,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 + f3[0,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 - f3[1,k]>=0)
                # self.opti.subject_to(self.u_s*f3[2,k]*0.7 + f3[1,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 - f4[0,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 + f4[0,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 - f4[1,k]>=0)
                # self.opti.subject_to(self.u_s*f4[2,k]*0.7 + f4[1,k]>=0)
                # unilateral constraint
                self.opti.subject_to(f1[2, k] >= 0)
                self.opti.subject_to(f2[2, k] >= 0)
                self.opti.subject_to(f3[2, k] >= 0)
                self.opti.subject_to(f4[2, k] >= 0)

            elif self.N1 <= k <= self.N1+self.N2-1:
                self.opti.subject_to(f1[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f2[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f3[:, k] == np.array([0.0, 0.0, 0.0]))
                self.opti.subject_to(f4[:, k] == np.array([0.0, 0.0, 0.0]))

            elif k > self.N1+self.N2-1:
                self.opti.subject_to(p_left_front[:,k] == self.p_left_front_final)
                self.opti.subject_to(p_right_front[:,k] == self.p_right_front_final)
                self.opti.subject_to(p_left_rear[:,k] == self.p_left_rear_final)
                self.opti.subject_to(p_right_rear[:,k] == self.p_right_rear_final)
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
        q_i = np.zeros((4, self.NX))
        for i in range(self.NX):
            q_i[:,i] = quaternion_slerp(self.q_init, self.q_final, float(i/self.NX))
        self.opti.set_initial(self.q, q_i)
        f_i = np.zeros((3, self.NU))
        for k in range(self.NU):
            if k < self.N1 or k > self.N1+self.N2-1:
                f_i[2,k] = self.model.mass*9.81/4
        self.opti.set_initial(f1, f_i)
        self.opti.set_initial(f2, f_i)
        self.opti.set_initial(f3, f_i)
        self.opti.set_initial(f4, f_i)

        p_left_front_initialize = np.zeros((3, self.NU))
        p_right_front_initialize = np.zeros((3, self.NU))
        p_left_rear_initialize = np.zeros((3, self.NU))
        p_right_rear_initialize = np.zeros((3, self.NU))
        for i in range(self.NU):
            if k < self.N1:
                p_left_front_initialize[:,k] = self.p_left_front_init
                p_right_front_initialize[:,k] = self.p_right_front_init
                p_left_rear_initialize[:,k] = self.p_left_rear_init
                p_right_rear_initialize[:,k] = self.p_right_rear_init
            elif self.N1 <= k <= self.N1+self.N2-1:
                p_left_front_initialize[:,k] = self.p_left_front_init + (k-self.N1)/self.N2*(self.p_left_front_final-self.p_left_front_init)
                p_right_front_initialize[:,k] = self.p_right_front_init + (k-self.N1)/self.N2*(self.p_right_front_final-self.p_right_front_init)
                p_left_rear_initialize[:,k] = self.p_left_rear_init + (k-self.N1)/self.N2*(self.p_left_rear_final-self.p_left_rear_init)
                p_right_rear_initialize[:,k] = self.p_right_rear_init + (k-self.N1)/self.N2*(self.p_right_rear_final-self.p_right_rear_init)
            else:
                p_left_front_initialize[:,k] = self.p_left_front_final
                p_right_front_initialize[:,k] = self.p_right_front_final
                p_left_rear_initialize[:,k] = self.p_left_rear_final
                p_right_rear_initialize[:,k] = self.p_right_rear_final
        self.opti.set_initial(p_left_front, p_left_front_initialize)
        self.opti.set_initial(p_right_front, p_right_front_initialize)
        self.opti.set_initial(p_left_rear, p_left_rear_initialize)
        self.opti.set_initial(p_right_rear, p_right_rear_initialize)

        # Optimization costs
        # unit_quaternion_cost = 0
        # for k in range(self.N):
        #     unit_quaternion_cost += sumsqr(q[:, k])-1
        # costs = sumsqr(f)
        # costs = sumsqr(diff(self.H.T)) + sumsqr(diff(self.L.T)) + 5*sumsqr(self.r) + sumsqr(diff(f1.T)) + sumsqr(diff(f2.T)) + sumsqr(self.rd) \
        #         + sumsqr(diff(f3.T)) + sumsqr(diff(f4.T)) + sumsqr(diff(f5.T)) + sumsqr(diff(f6.T)) + sumsqr(diff(f7.T)) + sumsqr(diff(f8.T)) \
        #         + sumsqr(self.dt) + sumsqr(diff(p_left.T)) + sumsqr(diff(p_right.T))
        ang_vel_final = angular_vel_quaterion_derivative(self.q[:,-1], self.qd[:,-1])
        ang_vel_final_2 = angular_vel_quaterion_derivative(self.q[:,-2], self.qd[:,-2])
        q_fin_cost = 50*sumsqr(self.q[:,-3:-1]-q_i[:,-3:-1])#+1*sumsqr(self.q[:, :-1] - q_i[:, :-1])# + + 100*sumsqr(ang_vel_final+ang_vel_final_2)
        effort_cost = sumsqr(diff(f1.T)) + sumsqr(diff(f2.T)) + sumsqr(diff(f3.T)) + sumsqr(diff(f4.T)) + 0.05*(sumsqr(diff(p_left_front.T)) + sumsqr(diff(p_left_rear.T))\
                + sumsqr(diff(p_right_front.T))+ sumsqr(diff(p_right_rear.T)))+sumsqr(diff(self.H.T)) #- 1000*sumsqr(self.qd[:,self.N1:self.N1+self.N2])#+ + 10*sumsqr(self.qd) ## sumsqr(self.r) + sumsqr(self.rd) ++ sumsqr(self.H)
        for i in range(self.N2):
            ang_vel = angular_vel_quaterion_derivative(self.q[:,self.N1+i], self.qd[:,self.N1+i])
            effort_cost -= 900*ang_vel[2]
        self.opti.minimize(1e-4*effort_cost+q_fin_cost)

        options = {"ipopt.max_iter": 2000, "ipopt.warm_start_init_point": "yes", "ipopt.hessian_approximation": "exact"}
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
                  solution.value(p_left_front),
                  solution.value(p_right_front),
                  solution.value(p_left_rear),
                  solution.value(p_right_rear))

        # self.trajectory_generation(solution.value(self.r),solution.value(self.rd), solution.value(self.q),solution.value(self.qd), solution.value(self.dt), solution.value(self.Hd),\
        #                         solution.value(self.L), solution.value(self.Ld), solution.value(p_left_front), solution.value(p_right_front), solution.value(p_left_rear),\
        #                         solution.value(p_right_rear), solution.value(f1), solution.value(f2), solution.value(f3), solution.value(f4), 400)

    def set_initial_solution(self):
        # dT_i = np.ones((1, 3)) * 0.025
        r_i = np.zeros((3, self.NX))
        for i in range(self.NX):
                r_i[:,i] = self.r_init + i/self.NX*(self.r_final - self.r_init)
        # self.opti.set_initial(self.U, U_i)
        # self.opti.set_initial(self.dt, dT_i)
        self.opti.set_initial(self.r, r_i)

    def quaternion_to_rotation_matrix(self, q):
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        R = np.array([[1 - 2 * (qy ** 2 + qz ** 2), 2 * qx * qy - 2 * qw * qz, 2 * qw * qy + 2 * qx * qz],
                      [2 * qx * qy + 2 * qw * qz, 1 - 2 * (qx ** 2 + qz ** 2), 2 * qy * qz - 2 * qw * qx],
                      [2 * qx * qz - 2 * qw * qy, 2 * qw * qx + 2 * qy * qz, 1 - 2 * (qx ** 2 + qy ** 2)]])
        return R

    def trajectory_generation(self, r, rd, q, qd, dt, Hd, L, Ld, p_left_front, p_right_front, p_left_rear, p_right_rear, f1,
                              f2, f3, f4, freq):
        dt_plot = np.ones(self.NU)
        dt_plot = np.concatenate(([0.0], dt_plot))
        dt_plot[1:self.N1] *= dt[0]
        dt_plot[self.N1:self.N1 + self.N2] *= dt[1]
        dt_plot[self.N1 + self.N2:] *= dt[2]
        dt_plot_interval = dt_plot
        for i in range(1, self.NX):
            dt_plot[i] += dt_plot[i - 1]
        dt_plot_control = dt_plot[1:]
        p_left_front = np.concatenate((p_left_front, np.reshape(self.p_left_front_final, (3, 1))), axis=1)
        p_right_front = np.concatenate((p_right_front, np.reshape(self.p_right_front_final, (3, 1))), axis=1)
        p_left_rear = np.concatenate((p_left_rear, np.reshape(self.p_left_rear_final, (3, 1))), axis=1)
        p_right_rear = np.concatenate((p_right_rear, np.reshape(self.p_right_rear_final, (3, 1))), axis=1)
        p_left_front_vel = np.diff(p_left_front)/(1/freq)
        p_left_front_vel = np.hstack((p_left_front_vel, p_left_front_vel[:,-1].reshape(3,1)))
        p_right_front_vel = np.diff(p_right_front)/(1/freq)
        p_right_front_vel = np.hstack((p_right_front_vel, p_right_front_vel[:,-1].reshape(3,1)))
        p_left_rear_vel = np.diff(p_left_rear)/(1/freq)
        p_left_rear_vel = np.hstack((p_left_rear_vel, p_left_rear_vel[:,-1].reshape(3,1)))
        p_right_rear_vel = np.diff(p_right_rear)/(1/freq)
        p_right_rear_vel = np.hstack((p_right_rear_vel, p_right_rear_vel[:,-1].reshape(3,1)))

        p_left_front_acc = np.diff(p_left_front_vel)/(1/freq)
        p_left_front_acc = np.hstack((p_left_front_acc, p_left_front_acc[:,-1].reshape(3,1)))
        p_right_front_acc = np.diff(p_right_front_vel)/(1/freq)
        p_right_front_acc = np.hstack((p_right_front_acc, p_right_front_acc[:,-1].reshape(3,1)))
        p_left_rear_acc = np.diff(p_left_rear_vel)/(1/freq)
        p_left_rear_acc = np.hstack((p_left_rear_acc, p_left_rear_acc[:,-1].reshape(3,1)))
        p_right_rear_acc = np.diff(p_right_rear_vel)/(1/freq)
        p_right_rear_acc = np.hstack((p_right_rear_acc, p_right_rear_acc[:,-1].reshape(3,1)))
        f1 = np.concatenate((f1, np.reshape(f1[:, -1], (3, 1))), axis=1)
        f2 = np.concatenate((f2, np.reshape(f2[:, -1], (3, 1))), axis=1)
        f3 = np.concatenate((f3, np.reshape(f3[:, -1], (3, 1))), axis=1)
        f4 = np.concatenate((f4, np.reshape(f4[:, -1], (3, 1))), axis=1)
        phase = np.ones(self.NX)
        # double support phase(1) / right support phase(1)
        phase[:self.N1] *= 1.0
        # flight phase
        phase[self.N1:self.N1 + self.N2] *= 3.0
        # double support phase
        phase[self.N1 + self.N2:] *= 1.0
        # np.concatenate((p_left, np.reshape(self.p_left_final, (3,1))), axis=1)
        sample_num = int(dt_plot[-1] / (1 / freq))
        comx = interp1d(dt_plot, r[0, :], kind='cubic')
        comy = interp1d(dt_plot, r[1, :], kind='cubic')
        comz = interp1d(dt_plot, r[2, :], kind='cubic')
        comdx = interp1d(dt_plot, rd[0, :], kind='quadratic')
        comdy = interp1d(dt_plot, rd[1, :], kind='quadratic')
        comdz = interp1d(dt_plot, rd[2, :], kind='quadratic')
        comddx = interp1d(dt_plot, Hd[0, :] / self.model.mass, kind='linear')
        comddy = interp1d(dt_plot, Hd[1, :] / self.model.mass, kind='linear')
        comddz = interp1d(dt_plot, Hd[2, :] / self.model.mass, kind='linear')
        comqx = interp1d(dt_plot, q[0,:], kind='linear')
        comqy = interp1d(dt_plot, q[1,:], kind='linear')
        comqz = interp1d(dt_plot, q[2,:], kind='linear')
        comqw = interp1d(dt_plot, q[3,:], kind='linear')
        comdqx = interp1d(dt_plot, qd[0,:], kind='linear')
        comdqy = interp1d(dt_plot, qd[1,:], kind='linear')
        comdqz = interp1d(dt_plot, qd[2,:], kind='linear')
        angx = interp1d(dt_plot, L[0, :] / self.model.mass, kind='linear')
        angy = interp1d(dt_plot, L[1, :] / self.model.mass, kind='linear')
        angz = interp1d(dt_plot, L[2, :] / self.model.mass, kind='linear')
        angdx = interp1d(dt_plot, Ld[0, :] / self.model.mass, kind='linear')
        angdy = interp1d(dt_plot, Ld[1, :] / self.model.mass, kind='linear')
        angdz = interp1d(dt_plot, Ld[2, :] / self.model.mass, kind='linear')
        lfrontfootx = interp1d(dt_plot, p_left_front[0, :], kind='linear')
        lfrontfooty = interp1d(dt_plot, p_left_front[1, :], kind='linear')
        lfrontfootz = interp1d(dt_plot, p_left_front[2, :], kind='linear')
        rfrontfootx = interp1d(dt_plot, p_right_front[0, :], kind='linear')
        rfrontfooty = interp1d(dt_plot, p_right_front[1, :], kind='linear')
        rfrontfootz = interp1d(dt_plot, p_right_front[2, :], kind='linear')
        lrearfootx = interp1d(dt_plot, p_left_rear[0, :], kind='linear')
        lrearfooty = interp1d(dt_plot, p_left_rear[1, :], kind='linear')
        lrearfootz = interp1d(dt_plot, p_left_rear[2, :], kind='linear')
        rrearfootx = interp1d(dt_plot, p_right_rear[0, :], kind='linear')
        rrearfooty = interp1d(dt_plot, p_right_rear[1, :], kind='linear')
        rrearfootz = interp1d(dt_plot, p_right_rear[2, :], kind='linear')

        lfrontfootx_vel = interp1d(dt_plot, p_left_front_vel[0, :], kind='linear')
        lfrontfooty_vel = interp1d(dt_plot, p_left_front_vel[1, :], kind='linear')
        lfrontfootz_vel = interp1d(dt_plot, p_left_front_vel[2, :], kind='linear')
        rfrontfootx_vel = interp1d(dt_plot, p_right_front_vel[0, :], kind='linear')
        rfrontfooty_vel = interp1d(dt_plot, p_right_front_vel[1, :], kind='linear')
        rfrontfootz_vel = interp1d(dt_plot, p_right_front_vel[2, :], kind='linear')
        lrearfootx_vel = interp1d(dt_plot, p_left_rear_vel[0, :], kind='linear')
        lrearfooty_vel = interp1d(dt_plot, p_left_rear_vel[1, :], kind='linear')
        lrearfootz_vel = interp1d(dt_plot, p_left_rear_vel[2, :], kind='linear')
        rrearfootx_vel = interp1d(dt_plot, p_right_rear_vel[0, :], kind='linear')
        rrearfooty_vel = interp1d(dt_plot, p_right_rear_vel[1, :], kind='linear')
        rrearfootz_vel = interp1d(dt_plot, p_right_rear_vel[2, :], kind='linear')

        lfrontfootx_acc = interp1d(dt_plot, p_left_front_acc[0, :], kind='linear')
        lfrontfooty_acc = interp1d(dt_plot, p_left_front_acc[1, :], kind='linear')
        lfrontfootz_acc = interp1d(dt_plot, p_left_front_acc[2, :], kind='linear')
        rfrontfootx_acc = interp1d(dt_plot, p_right_front_acc[0, :], kind='linear')
        rfrontfooty_acc = interp1d(dt_plot, p_right_front_acc[1, :], kind='linear')
        rfrontfootz_acc = interp1d(dt_plot, p_right_front_acc[2, :], kind='linear')
        lrearfootx_acc = interp1d(dt_plot, p_left_rear_acc[0, :], kind='linear')
        lrearfooty_acc = interp1d(dt_plot, p_left_rear_acc[1, :], kind='linear')
        lrearfootz_acc = interp1d(dt_plot, p_left_rear_acc[2, :], kind='linear')
        rrearfootx_acc = interp1d(dt_plot, p_right_rear_acc[0, :], kind='linear')
        rrearfooty_acc = interp1d(dt_plot, p_right_rear_acc[1, :], kind='linear')
        rrearfootz_acc = interp1d(dt_plot, p_right_rear_acc[2, :], kind='linear')

        f1x = interp1d(dt_plot, f1[0, :], kind='linear')
        f1y = interp1d(dt_plot, f1[1, :], kind='linear')
        f1z = interp1d(dt_plot, f1[2, :], kind='linear')
        f2x = interp1d(dt_plot, f2[0, :], kind='linear')
        f2y = interp1d(dt_plot, f2[1, :], kind='linear')
        f2z = interp1d(dt_plot, f2[2, :], kind='linear')
        f3x = interp1d(dt_plot, f3[0, :], kind='linear')
        f3y = interp1d(dt_plot, f3[1, :], kind='linear')
        f3z = interp1d(dt_plot, f3[2, :], kind='linear')
        f4x = interp1d(dt_plot, f4[0, :], kind='linear')
        f4y = interp1d(dt_plot, f4[1, :], kind='linear')
        f4z = interp1d(dt_plot, f4[2, :], kind='linear')
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
        sample_comqx = comqx(sample_t)
        sample_comqy = comqy(sample_t)
        sample_comqz = comqz(sample_t)
        sample_comqw = comqw(sample_t)
        sample_comdqx = comdqx(sample_t)
        sample_comdqy = comdqy(sample_t)
        sample_comdqz = comdqz(sample_t)
        sample_angx = angx(sample_t)
        sample_angy = angy(sample_t)
        sample_angz = angz(sample_t)
        sample_angdx = angdx(sample_t)
        sample_angdy = angdy(sample_t)
        sample_angdz = angdz(sample_t)
        sample_lfrontfootx = lfrontfootx(sample_t)
        sample_lfrontfooty = lfrontfooty(sample_t)
        sample_lfrontfootz = lfrontfootz(sample_t)
        sample_rfrontfootx = rfrontfootx(sample_t)
        sample_rfrontfooty = rfrontfooty(sample_t)
        sample_rfrontfootz = rfrontfootz(sample_t)
        sample_lrearfootx = lrearfootx(sample_t)
        sample_lrearfooty = lrearfooty(sample_t)
        sample_lrearfootz = lrearfootz(sample_t)
        sample_rrearfootx = rrearfootx(sample_t)
        sample_rrearfooty = rrearfooty(sample_t)
        sample_rrearfootz = rrearfootz(sample_t)

        sample_lfrontfootx_vel = lfrontfootx_vel(sample_t)
        sample_lfrontfooty_vel = lfrontfooty_vel(sample_t)
        sample_lfrontfootz_vel = lfrontfootz_vel(sample_t)
        sample_rfrontfootx_vel = rfrontfootx_vel(sample_t)
        sample_rfrontfooty_vel = rfrontfooty_vel(sample_t)
        sample_rfrontfootz_vel = rfrontfootz_vel(sample_t)
        sample_lrearfootx_vel = lrearfootx_vel(sample_t)
        sample_lrearfooty_vel = lrearfooty_vel(sample_t)
        sample_lrearfootz_vel = lrearfootz_vel(sample_t)
        sample_rrearfootx_vel = rrearfootx_vel(sample_t)
        sample_rrearfooty_vel = rrearfooty_vel(sample_t)
        sample_rrearfootz_vel = rrearfootz_vel(sample_t)

        sample_lfrontfootx_acc = lfrontfootx_acc(sample_t)
        sample_lfrontfooty_acc = lfrontfooty_acc(sample_t)
        sample_lfrontfootz_acc = lfrontfootz_acc(sample_t)
        sample_rfrontfootx_acc = rfrontfootx_acc(sample_t)
        sample_rfrontfooty_acc = rfrontfooty_acc(sample_t)
        sample_rfrontfootz_acc = rfrontfootz_acc(sample_t)
        sample_lrearfootx_acc = lrearfootx_acc(sample_t)
        sample_lrearfooty_acc = lrearfooty_acc(sample_t)
        sample_lrearfootz_acc = lrearfootz_acc(sample_t)
        sample_rrearfootx_acc = rrearfootx_acc(sample_t)
        sample_rrearfooty_acc = rrearfooty_acc(sample_t)
        sample_rrearfootz_acc = rrearfootz_acc(sample_t)
        sample_f1x = f1x(sample_t)
        sample_f1y = f1y(sample_t)
        sample_f1z = f1z(sample_t)
        sample_f2x = f2x(sample_t)
        sample_f2y = f2y(sample_t)
        sample_f2z = f2z(sample_t)
        sample_f3x = f3x(sample_t)
        sample_f3y = f3y(sample_t)
        sample_f3z = f3z(sample_t)
        sample_f4x = f4x(sample_t)
        sample_f4y = f4y(sample_t)
        sample_f4z = f4z(sample_t)
        sample_phase = phase_t(sample_t)
        sample_COM = np.vstack((sample_comx, sample_comy, sample_comz))
        sample_dCOM = np.vstack((sample_comdx, sample_comdy, sample_comdz))
        sample_ddCOM = np.vstack((sample_comddx, sample_comddy, sample_comddz))
        sample_COM_ori = np.vstack((sample_comqx,sample_comqy,sample_comqz,sample_comqw))
        sample_qCOM_ori = np.vstack((sample_comdqx,sample_comdqy,sample_comdqz))
        sample_Ang = np.vstack((sample_angx, sample_angy, sample_angz))
        sample_dAng = np.vstack((sample_angdx, sample_angdy, sample_angdz))
        sample_lfrontFOOT = np.vstack((sample_lfrontfootx, sample_lfrontfooty, sample_lfrontfootz))
        sample_rfrontFOOT = np.vstack((sample_rfrontfootx, sample_rfrontfooty, sample_rfrontfootz))
        sample_lrearFOOT = np.vstack((sample_lrearfootx, sample_lrearfooty, sample_lrearfootz))
        sample_rrearFOOT = np.vstack((sample_rrearfootx, sample_rrearfooty, sample_rrearfootz))
        sample_lfrontFOOT_vel = np.vstack((sample_lfrontfootx_vel, sample_lfrontfooty_vel, sample_lfrontfootz_vel))
        sample_rfrontFOOT_vel = np.vstack((sample_rfrontfootx_vel, sample_rfrontfooty_vel, sample_rfrontfootz_vel))
        sample_lrearFOOT_vel = np.vstack((sample_lrearfootx_vel, sample_lrearfooty_vel, sample_lrearfootz_vel))
        sample_rrearFOOT_vel = np.vstack((sample_rrearfootx_vel, sample_rrearfooty_vel, sample_rrearfootz_vel))
        sample_lfrontFOOT_acc = np.vstack((sample_lfrontfootx_acc, sample_lfrontfooty_acc, sample_lfrontfootz_acc))
        sample_rfrontFOOT_acc = np.vstack((sample_rfrontfootx_acc, sample_rfrontfooty_acc, sample_rfrontfootz_acc))
        sample_lrearFOOT_acc = np.vstack((sample_lrearfootx_acc, sample_lrearfooty_acc, sample_lrearfootz_acc))
        sample_rrearFOOT_acc = np.vstack((sample_rrearfootx_acc, sample_rrearfooty_acc, sample_rrearfootz_acc))
        sample_f1 = np.vstack((sample_f1x, sample_f1y, sample_f1z))
        sample_f2 = np.vstack((sample_f2x, sample_f2y, sample_f2z))
        sample_f3 = np.vstack((sample_f3x, sample_f3y, sample_f3z))
        sample_f4 = np.vstack((sample_f4x, sample_f4y, sample_f4z))
        for i, v in enumerate(sample_phase):
            if v <= 2.0:
                sample_phase[i] = 1
            else:
                # double support is 0, flight phase is 1, right support is 1, left support is -1
                sample_phase[i] = 0
                # elif 2.0 < v < 3.0:
                #     sample_phase[i] = 0
        # fig, axs = plt.subplots(2, 2)
        # # axs = fig.gca(projection='3d')
        # axs[0][0].plot(sample_COM.T)
        # axs[0][0].plot(sample_phase.T)
        # axs[0][0].legend(['rx', 'ry', 'rz'])
        # axs[0][1].plot(dt_plot)
        # axs[0][1].legend(['dt'])

        plt.plot(sample_comqx)
        plt.plot(sample_comqy)
        plt.plot(sample_comqz)
        plt.plot(sample_comqw)
        plt.show()

        np.savez('/home/robin/Documents/anymal_ctrl_edin_gaits/src/anymal_control_msg_publisher/scripts/data_test_backflip.npz',
            lfront=sample_lfrontFOOT, rfront=sample_rfrontFOOT, lrear=sample_lrearFOOT, \
            rrear=sample_rrearFOOT, lfront_vel=sample_lfrontFOOT_vel, rfront_vel=sample_rfrontFOOT_vel, lrear_vel=sample_lrearFOOT_vel, \
            rrear_vel=sample_rrearFOOT_vel, lfront_acc=sample_lfrontFOOT_acc, rfront_acc=sample_rfrontFOOT_acc, lrear_acc=sample_lrearFOOT_acc, \
            rrear_acc=sample_rrearFOOT_acc, f1=sample_f1, f2=sample_f2, f3=sample_f3, f4=sample_f4, time=sample_t, \
            com=sample_COM, dcom=sample_dCOM, ddcom=sample_ddCOM, com_ori=sample_COM_ori, com_ori_vel=sample_qCOM_ori, ang=sample_Ang, dang=sample_dAng, phase=sample_phase)

    def plot(self, r, q, H, L, dt, rd, qd, Hd, Ld, f1, f2, f3, f4, p1, p2, p3, p4):#
        dt_plot = np.ones(self.NU)
        dt_plot = np.concatenate(([0.0], dt_plot))
        dt_plot[1:self.N1] *= dt[0]
        dt_plot[self.N1:self.N1+self.N2] *= dt[1]
        dt_plot[self.N1+self.N2:] *= dt[2]
        rpy = np.zeros((3, self.NX))
        ang_vel = np.zeros((3, self.NX))
        for i in range(self.NX):
            rpy[0, i], rpy[1, i], rpy[2, i] = euler_from_quaternion(q[:, i])
            ang_vel[:, i] = angular_vel_quaterion_derivative_numerical(q[:, i], qd[:, i])
        for i in range(1, self.NX):
            dt_plot[i] += dt_plot[i - 1]

        dt_plot_control = dt_plot[1:]
        fig, axs = plt.subplots(3, 2)
        # axs = fig.gca(projection='3d')
        axs[0][0].plot(dt_plot, r.T)
        axs[0][0].legend(['rx', 'ry', 'rz'])

        # axs[0][1].plot(dt_plot, q.T)
        # axs[0][1].legend(['qx', 'qy', 'qz', 'qw'])
        axs[0][1].plot(dt_plot, rpy.T)
        axs[0][1].legend(['r', 'p', 'y'])

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

        axs[0][1].plot(dt_plot, ang_vel.T)
        # axs[0][1].plot(np.linalg.norm(qd, axis=0))
        axs[0][1].legend(['wx', 'wy', 'wz'])
        rpy_flight_yaw = ang_vel[2, self.N1:self.N1+self.N2]
        print('The average velocity is ', np.mean(rpy_flight_yaw))

        axs[1][0].plot(dt_plot, Hd.T)
        axs[1][0].legend(['dHx', 'dHy', 'dHz'])

        axs[1][1].plot(dt_plot, Ld.T)
        axs[1][1].legend(['dLx', 'dLy', 'dLz'])

        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(dt_plot_control, f1.T)
        axs[0][0].legend(['f_left_front_x', 'f_left_front_y', 'f_left_front_z'])

        axs[1][0].plot(dt_plot_control, f2.T)
        axs[1][0].legend(['f_right_front_x', 'f_right_front_y', 'f_right_front_z'])

        axs[0][1].plot(dt_plot_control, f3.T)
        axs[0][1].legend(['f_left_rear_x', 'f_left_rear_y', 'f_left_rear_z'])

        axs[1][1].plot(dt_plot_control, f4.T)
        axs[1][1].legend(['f_right_rear_x', 'f_right_rear_y', 'f_right_rear_z'])

        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(dt_plot_control, p1.T)
        axs[0][0].legend(['p_left_front_x', 'p_left_front_y', 'p_left_front_z'])
        #
        axs[1][0].plot(dt_plot_control, p2.T)
        axs[1][0].legend(['p_right_front_x', 'p_right_front_y', 'p_right_front_z'])

        axs[0][1].plot(dt_plot_control, p3.T)
        axs[0][1].legend(['p_left_rear_x', 'p_left_rear_y', 'p_left_rear_z'])
        #
        axs[1][1].plot(dt_plot_control, p4.T)
        axs[1][1].legend(['p_right_rear_x', 'p_right_rear_y', 'p_right_rear_z'])

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
        ax.set_xlim3d([-0.8, 0.8])
        ax.set_ylim3d([-0.8, 0.8])
        ax.set_zlim3d([-0.05, 1.0])
        # ax.set_aspect('equal')
        # ax.plot3D(p1[0,0].T, p1[1,0].T, p1[2,0].T, 'ro')
        p1_value_x = [p1[0,0],p2[0,0]]
        p1_value_y = [p1[1,0],p2[1,0]]
        p1_value_z = [p1[2,0],p2[2,0]]
        p2_value_x = [p2[0,0],p3[0,0]]
        p2_value_y = [p2[1,0],p3[1,0]]
        p2_value_z = [p2[2,0],p3[2,0]]
        p3_value_x = [p3[0,0],p4[0,0]]
        p3_value_y = [p3[1,0],p4[1,0]]
        p3_value_z = [p3[2,0],p4[2,0]]
        # ax.plot(p1_value_x, p1_value_y, p1_value_z)
        # ax.plot(p2_value_x, p2_value_y, p2_value_z)
        # ax.plot(p3_value_x, p3_value_y, p3_value_z)
        ax.plot3D(p1[0,:].T, p1[1,:].T, p1[2,:].T, 'gray')
        ax.plot3D(p2[0,:].T, p2[1,:].T, p2[2,:].T, 'gray')
        ax.plot3D(p3[0,:].T, p3[1,:].T, p3[2,:].T, 'gray')
        ax.plot3D(p4[0,:].T, p4[1,:].T, p4[2,:].T, 'gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        print('The landing angle is ', rpy[:, self.N1+self.N2])
        print('The time of phase 1 is ', dt_plot[self.N1-1])
        print('The time of phase 2 is ', dt_plot[self.N1+self.N2])

        plt.show()

if __name__ == "__main__":
    single_rigid_body_model = SingleRigidBodyModel()
    motion_planner = MotionPlanner(model=single_rigid_body_model)
