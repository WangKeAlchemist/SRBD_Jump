import pyquaternion
import numpy as np
import utils.transformations as tf


def quaternionIntegration(q, omege, dt):
    '''
    Quaternion integration from angular velocity for given time step.
    :param q: initial quaternion
    :param w: angular velocity
    :param dt: time step
    :return: new quaternion with one step integration
    '''
    w_hat = np.array([omege[0], omege[1], omege[2], 0])
    # q_dot = 0.5 * tf.quaternion_multiply(q, w_hat)
    q_dot = 0.5 * tf.quaternion_multiply(w_hat, q)
    dq = q_dot*dt
    q = q+dq
    q = q/np.linalg.norm(q)
    return q


def quaternionIntegration2(q, omege, dt):
    '''
    Quaternion integration from angular velocity for given time step.
    :param q: initial quaternion
    :param w: angular velocity
    :param dt: time step
    :return: new quaternion with one step integration
    '''
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    Q = np.array([[qw, -qz, qy],
                  [qz, qw, -qx],
                  [-qy, qx, qw],
                  [-qx, -qy, -qz]])

    q_dot = 1/2 * Q @ omege

    dq = q_dot*dt
    q = q+dq
    q = q/np.linalg.norm(q)
    return q


def testQuaternionIntegration1():
    omega = np.array([np.pi / 2, np.pi/2, np.pi/2])
    dt = 0.001
    N = round(1/dt)
    q = np.array([0,0,0,1])

    for i in range(N):
        q = quaternionIntegration(q, omega, dt)
    print('xyzw:',q)

    # np.testing.assert_almost_equal(q, tf.quaternion_from_euler(0, np.pi/2, np.pi/2))
    # print('testQuaternionIntegration() pass!')

def testQuaternionIntegration2():
    omega = np.array([np.pi / 2, np.pi/2, np.pi/2])
    dt = 0.001
    N = round(1/dt)
    q = np.array([0,0,0,1])

    for i in range(N):
        q = quaternionIntegration2(q, omega, dt)
    print('xyzw:',q)

    # np.testing.assert_almost_equal(q, tf.quaternion_from_euler(0, np.pi/2, np.pi/2))
    # print('testQuaternionIntegration() pass!')


def testQuaternionIntegration():
    omega = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
    dt = 0.001
    N = round(1 / dt)
    q = pyquaternion.Quaternion([1,0,0,0])

    for i in range(N):
        q.integrate(omega, dt)
    print('wxyz:', q)


if __name__ == "__main__":
    testQuaternionIntegration()
    testQuaternionIntegration1()
    testQuaternionIntegration2()

    # q1 = tf.quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    # print(q1)
    # q2 = tf.quaternion_multiply([-5, 6, 7, 8], [1, -2, 3, 4])
    # print(q2)
    #
    #
    # print('*'*100)
    #
    # q1 = np.array([-2, 3, 4, 1])
    # q2 = np.array([6, 7, 8, -5])
    # q1 = q1 / np.linalg.norm(q1)
    # q2 = q2 / np.linalg.norm(q2)
    # print(tf.quaternion_multiply(q1, q2))
    # print(tf.quaternion_multiply(q2, q1))
    #
    # q1 = pyquaternion.Quaternion([1, -2, 3, 4])
    # q2 = pyquaternion.Quaternion([-5, 6, 7, 8])
    # print(q1*q2)
    # print(q2*q1)


