import numpy as np
from casadi import *
import utils.transformations as tf
from scipy.interpolate import interp1d

class SingleRigidBodyModel:
    def __init__(self, mass=13.8, size=(0.4, 0.2, 0.1), g=(0, 0, -9.81)):
        self.mass = mass
        self.size = size
        self.g = np.array(g)
        self.I_body = self.local_inertia_tensor(mass, size)
        self.I_body_inv = np.linalg.inv(self.I_body)


    def dynamics(self, r, q, v, w, f, tau):
        rd = v
        R = self.quaternion_to_rotation_matrix(q)
        I = R @ self.I_body @ R.T
        I_inv = R @ self.I_body_inv @ R.T
        qd = 0.5 * self.w_matrix(w) @ q
        vd = f/self.mass + self.g
        wd = I_inv@(tau - cross(w, I@w))
        return rd, qd, vd, wd


    def Q(self, quaternion):
        qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        Q = blockcat([[qw, -qz, qy],
                      [qz, qw, -qx],
                      [-qy, qx, qw],
                      [-qx, -qy, -qz]])
        return Q

    def w_matrix(self, w):
        wx, wy, wz = w[0], w[1], w[2]
        W = blockcat([[0, wz, -wy, wx],
                      [-wz, 0, wx, wy],
                      [wy, -wx, 0, wz],
                      [-wx, -wy, -wz, 0]])
        return W

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
    def __init__(self, model, T=0.5, N=100, nlp_solver='ipopt'):
        self.model = model

        # define parameters
        self.T = T
        self.N = N
        self.NX = self.N + 1
        self.NU = self.N
        self.dt = self.T/self.N
        print('T:', self.T)
        print('N:', self.N)
        print('dt:', self.dt)

        # Optimization solver
        self.opti = Opti()
        self.opti.solver(nlp_solver)

        # Optimization variables
        # states
        r = self.opti.variable(3, self.NX)  # body position
        q = self.opti.variable(4, self.NX)  # body quaternion
        v = self.opti.variable(3, self.NX)  # body linear velocity
        w = self.opti.variable(3, self.NX)  # body angular velocity

        # inputs
        f = self.opti.variable(3, self.NU)
        tau = self.opti.variable(3, self.NU)

        # ini and fin states
        r_ini = np.array([0,0.1,0])
        q_ini = np.array([0,0,0,1])
        v_ini = np.zeros(3)
        w_ini = np.zeros(3)

        r_fin = np.array([0.0,0.0,0.0])
        q_fin = tf.quaternion_from_euler(0, 0, np.pi/2)
        v_fin = np.zeros(3)
        w_fin = np.zeros(3)


        # Optimization costs
        q_fin_cost = sumsqr(q[:, -1] - q_fin)
        w_fin_cost = sumsqr(w[:, -1] - w_fin)
        tau_cost = sumsqr(tau)
        self.opti.minimize(q_fin_cost + w_fin_cost + 1e-6*tau_cost)

        # dynamic constraints
        for k in range(self.N):
            rd, qd, vd, wd = self.model.dynamics(r[:, k], q[:, k], v[:, k], w[:, k], f[:, k], tau[:, k])
            self.opti.subject_to(r[:, k + 1] == r[:, k] + rd * self.dt)
            self.opti.subject_to(q[:, k + 1] == q[:, k] + qd * self.dt)
            self.opti.subject_to(v[:, k + 1] == v[:, k] + vd * self.dt)
            self.opti.subject_to(w[:, k + 1] == w[:, k] + wd * self.dt)

        # # unit quaternion constraints
        # for k in range(self.N):
        #     self.opti.subject_to(sumsqr(q[:, k]) == 1)


        # boundary constraints
        self.opti.subject_to(r[:, 0] == r_ini)
        self.opti.subject_to(q[:, 0] == q_ini)
        self.opti.subject_to(v[:, 0] == v_ini)
        self.opti.subject_to(w[:, 0] == w_ini)

        self.opti.subject_to(r[:, -1] == r_fin)
        self.opti.subject_to(v[:, -1] == v_fin)
        # self.opti.subject_to(q[:, -1] == q_fin)
        self.opti.subject_to(w[:, -1] == w_fin)
        # self.opti.subject_to(L[:, -1] == L_fin)

        # set initial value
        # time = np.linspace(0,self.T,self.NX)
        # fun_r = interp1d(np.array([0, self.T]), np.vstack((r_ini, r_fin)), axis=0)
        # r_initial = fun_r(time)
        # self.opti.set_initial(r, r_initial.T)


        q_slerp = np.array([tf.quaternion_slerp(q_ini, q_fin, i / self.NX) for i in range(self.NX)])
        self.opti.set_initial(q, q_slerp.T)

        # solve the optimization problem
        solution = self.opti.solve()

        # plot
        self.plot(solution.value(r),
                  solution.value(q),
                  solution.value(v),
                  solution.value(w),
                  solution.value(f),
                  solution.value(tau),
                  q_slerp)

    def plot(self, r, q, v, w, f, tau, q_slerp):

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 3)
        axs[0][0].plot(r.T)
        axs[0][0].legend(['rx','ry','rz'])

        axs[0][1].plot(q.T)
        axs[0][1].plot(np.linalg.norm(q,axis=0),'--')
        axs[0][1].legend(['qx', 'qy', 'qz', 'qw', 'norm'])
        axs[0][1].grid()

        # euler_angle = np.array([tf.euler_from_quaternion(qi) for qi in q.T])
        # axs[0][2].plot(np.rad2deg(euler_angle))
        # axs[0][2].legend(['roll', 'pitch', 'yaw'])
        axs[0][2].plot(q_slerp)
        axs[0][2].legend(['slerp_qx', 'slerp_qy', 'slerp_qz', 'slerp_qw'])


        axs[1][0].plot(v.T)
        axs[1][0].legend(['vx','vy','vz'])

        axs[1][1].plot(w.T)
        axs[1][1].legend(['wx','wy','wz'])


        axs[2][0].plot(f.T)
        axs[2][0].legend(['fx','fy','fz'])

        axs[2][1].plot(tau.T)
        axs[2][1].legend(['taux','tauy','tauz'])


        plt.show()

if __name__ == "__main__":
    single_rigid_body_model = SingleRigidBodyModel()
    motion_planner = MotionPlanner(model=single_rigid_body_model)
