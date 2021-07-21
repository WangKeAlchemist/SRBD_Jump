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
    def __init__(self, mass=1, g=(-9.81)):
        self.mass = mass
        self.g = np.array(g)

    def simple_dynamics(self, r, f, H):
        rd = H / self.mass # velocity
        Hd = f + self.mass * self.g # acceleration
        return rd, Hd

class MotionPlanner:
    def __init__(self, model, T=1, N=50, nlp_solver='ipopt'):
        self.model = model

        # define parameters
        self.T = T
        # self.N = N
        self.N1 = 15

        # dt = 0.05
        self.N = self.N1
        self.NX = self.N + 1
        self.NU = self.N
        # self.dt = self.T/self.N
        # side jump: 1.05
        self.r_init = np.array([0.0])
        self.r_final = np.array([0.5])
        self.u_s = 1.0 # friction coefficient

        # Optimization solver
        self.opti = Opti()
        self.opti.solver(nlp_solver)

        # Optimization variables
        # states
        self.r = self.opti.variable(self.NX)  # body position
        self.H = self.opti.variable(self.NX)  # body linear momentum

        self.rd = self.opti.variable(self.NX)  # body position
        self.Hd = self.opti.variable(self.NX)  # body linear momentum

        # inputs
        # self.dt = self.opti.variable(1)
        self.dt = 0.02
        self.f = self.opti.variable(self.NU)

        # Optimization costs
        costs = sumsqr(self.f*self.rd[:-1]) #+ sumsqr(p)

        self.opti.minimize(costs)

        # dynamic constraints
        for k in range(self.N1):
            self.rd[k], self.Hd[k] = self.model.simple_dynamics(self.r[k], self.H[k], self.f[k])
            self.opti.subject_to(self.r[k + 1] == self.r[k] + self.rd[k] * self.dt)
            self.opti.subject_to(self.H[k + 1] == self.H[k] + self.Hd[k] * self.dt)

        # unit quaternion constraints

        # for j in range(3):
        # self.opti.subject_to(self.dt >= 0.005)
        # self.opti.subject_to(self.dt <= 0.5)

        # constraints on contact positions
        # boundary constraints
        self.opti.subject_to(self.r[0] == self.r_init)
        self.opti.subject_to(self.H[0] == np.array([0]))

        self.opti.subject_to(self.r[-1] == self.r_final)
        # self.opti.subject_to(q[:, -1] == tf.quaternion_from_euler(0.0,0.01,0.0))
        self.opti.subject_to(self.H[-1] == np.array([0]))
        # self.opti.subject_to(L[:, -1] == np.array([0,0.1,0]))

        # self.set_initial_solution()

        options = {"ipopt.max_iter": 20000, "ipopt.warm_start_init_point": "yes"}
        # "verbose": True, "ipopt.print_level": 0, "print_out": False, "print_in": False, "print_time": False,"ipopt.hessian_approximation": "limited-memory",
        self.opti.solver("ipopt", options)
        solution = self.opti.solve()

        # plot
        # self.plot(solution.value(self.r),
        #           solution.value(self.H),
        #           solution.value(self.dt),
        #           solution.value(self.rd),
        #           solution.value(self.Hd),
        #           solution.value(self.f))

        # self.trajectory_generation(solution.value(self.r),solution.value(self.rd),solution.value(p_left), solution.value(p_right), solution.value(self.dt), solution.value(self.Hd))

    def set_initial_solution(self):
        dT_i = np.ones((1, 3)) * 0.1
        r_i = np.zeros((3, self.NX))
        for i in range(self.NX):
                r_i[:,i] = self.r_init + i/self.NX*(self.r_final - self.r_init)
        # self.opti.set_initial(self.U, U_i)
        self.opti.set_initial(self.dt, dT_i)
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
        # double support phase
        phase[:self.N1] *= 0.0
        # flight phase
        phase[self.N1:self.N1+self.N2] *= 3.0
        # double support phase
        phase[self.N1+self.N2:] *= 0.0
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
        lfootx = interp1d(dt_plot, p_left[0,:], kind='quadratic')
        lfooty = interp1d(dt_plot, p_left[1,:], kind='quadratic')
        lfootz = interp1d(dt_plot, p_left[2,:], kind='quadratic')
        rfootx = interp1d(dt_plot, p_right[0,:], kind='quadratic')
        rfooty = interp1d(dt_plot, p_right[1,:], kind='quadratic')
        rfootz = interp1d(dt_plot, p_right[2,:], kind='quadratic')
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
            if v > 0:
                sample_phase[i] = 3
        fig, axs = plt.subplots(2, 2)
        # axs = fig.gca(projection='3d')
        axs[0][0].plot(sample_COM.T)
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
        np.savez('data_forward.npz', copl=sample_lFOOT, copr=sample_rFOOT, time=sample_t, com=sample_COM, dcom=sample_dCOM, ddcom=sample_ddCOM, phase=sample_phase)
        plt.show()

    def plot(self, r, q, H, L, dt, rd, qd, Hd, Ld, f):
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
        axs[0][1].plot(np.linalg.norm(q, axis=0))
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

        plt.show()

if __name__ == "__main__":
    single_rigid_body_model = SingleRigidBodyModel()
    motion_planner = MotionPlanner(model=single_rigid_body_model)
