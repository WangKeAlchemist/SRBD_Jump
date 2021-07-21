import matplotlib.pyplot as plt
import  numpy as np
import utils.transformations as tf


if __name__ == "__main__":

#################### JUMP UP ####################################
    # with np.load('data_forward_up_30_higher_2.npz') as ref_data:
    #     copl = ref_data['copl']
    #     copr = ref_data['copr']
    #     time_sample = ref_data['time']
    #     com =  ref_data['com']
    #     com_vel = ref_data['dcom']
    #     com_acc = ref_data['ddcom']
    #     phase = ref_data['phase']
    #
    # with np.load('data_com_jumpUp.npz') as ref_data:
    #     com_real = ref_data['com']
    #     com_vel_real = ref_data['com_vel']
    #     copl_real = ref_data['copl']
    #     copr_real = ref_data['copr']
    #
    # with np.load('data_lslide_jumpUp.npz') as ref_data:
    #     lslide = ref_data['lside']
    # with np.load('data_rslide_jumpUp.npz') as ref_data:
    #     rslide = ref_data['rside']

    # print('mean is ', np.mean(abs(lslide)))
    # print('max is ', np.max(abs(lslide)))
    #
    #
    #
    # copr[2, :] *= 1.15  # not right when jump in stage
    # copl[2, :] *= 1.15
    # copl_real[2,:] -= 0.05
    # copr_real[2,:] -= 0.05
    # start = float(np.where(phase > 0)[0][0]/1000)
    # end = float(np.where(phase > 0)[0][-1]/1000)
    # plt.rc('xtick', labelsize=30)
    # plt.rc('ytick', labelsize=30)
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(time_sample,lslide,linewidth=5)
    # ax.plot(time_sample,rslide,linewidth=5)
    # plt.axvline(x=start, color='r', linestyle='--', linewidth=5)
    # plt.axvline(x=end, color='r', linestyle='--', linewidth=5)
    # plt.xlabel('t [s]', fontsize=45)
    # plt.ylabel('[N.m]', fontsize=45)
    # ax.legend(['Left Slide', 'Right Slide'], prop=dict(size='45'))
    # ax.set_title('SLIDE Torque', fontweight='bold', fontsize=55)
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(time_sample, com[0,:], linewidth=5)
    # ax.plot(time_sample, com[1,:], linewidth=5)
    # ax.plot(time_sample, com[2,:], linewidth=5)
    # ax.plot(time_sample, com_real[0,:], linestyle='--', linewidth=5)
    # ax.plot(time_sample, com_real[1,:], linestyle='--',linewidth=5)
    # ax.plot(time_sample, com_real[2,:], linestyle='--',linewidth=5)
    # plt.axvline(x=start, color='r', linestyle='--', linewidth=5)
    # plt.axvline(x=end, color='r', linestyle='--', linewidth=5)
    # plt.xlabel('$\mathbf{t} \hspace{0.5cm} \mathbf{[s]}$', fontsize=45)
    # plt.ylabel('$\mathbf{[m]}$', fontsize=45)
    # ax.legend(['x', 'y', 'z'], prop=dict(size='45'))
    # ax.set_title("$\mathbf{CoM} \hspace{1cm} \mathbf{linear} \hspace{1cm} \mathbf{position}$", fontweight='bold', fontsize=55)
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(time_sample, com_vel[0,:], linewidth=5)
    # ax.plot(time_sample, com_vel[1,:], linewidth=5)
    # ax.plot(time_sample, com_vel[2,:], linewidth=5)
    # ax.plot(time_sample, com_vel_real[0,:], linestyle='--', linewidth=5)
    # ax.plot(time_sample, com_vel_real[1,:], linestyle='--',linewidth=5)
    # ax.plot(time_sample, com_vel_real[2,:], linestyle='--',linewidth=5)
    # plt.axvline(x=start, color='r', linestyle='--', linewidth=5)
    # plt.axvline(x=end, color='r', linestyle='--', linewidth=5)
    # plt.xlabel('$\mathbf{t} \hspace{0.5cm} \mathbf{[s]}$', fontsize=45)
    # plt.ylabel('$\mathbf{[m/s]}$',  fontsize=45)
    # ax.legend(['x', 'y', 'z'], prop=dict(size='45'))
    # ax.set_title('$\mathbf{CoM} \hspace{1cm} \mathbf{linear} \hspace{1cm} \mathbf{velocity}$', fontweight='bold', fontsize=55)
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(time_sample, copl[0,:], linewidth=5)
    # ax.plot(time_sample, copl[1,:], linewidth=5)
    # ax.plot(time_sample, copl[2,:], linewidth=5)
    # ax.plot(time_sample, copl_real[0,:], linestyle='--', linewidth=5)
    # ax.plot(time_sample, copl_real[1,:], linestyle='--',linewidth=5)
    # ax.plot(time_sample, copl_real[2,:], linestyle='--',linewidth=5)
    # plt.axvline(x=start, color='r', linestyle='--', linewidth=5)
    # plt.axvline(x=end, color='r', linestyle='--', linewidth=5)
    # plt.xlabel('$\mathbf{t} \hspace{0.5cm} \mathbf{[s]}$', fontsize=45)
    # plt.ylabel('$\mathbf[m]$',  fontsize=45)
    # ax.legend(['x', 'y', 'z'], prop=dict(size='45'))
    # ax.set_title('$\mathbf{Left} \hspace{1cm} \mathbf{foot} \hspace{1cm} \mathbf{position}$', fontweight='bold', fontsize=55)
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(time_sample, copr[0,:], linewidth=5)
    # ax.plot(time_sample, copr[1,:], linewidth=5)
    # ax.plot(time_sample, copr[2,:], linewidth=5)
    # ax.plot(time_sample, copr_real[0,:], linestyle='--', linewidth=5)
    # ax.plot(time_sample, copr_real[1,:], linestyle='--',linewidth=5)
    # ax.plot(time_sample, copr_real[2,:], linestyle='--',linewidth=5)
    # plt.axvline(x=start, color='r', linestyle='--', linewidth=5)
    # plt.axvline(x=end, color='r', linestyle='--', linewidth=5)
    # plt.xlabel('$\mathbf{t} \hspace{0.5cm} \mathbf{[s]}$', fontsize=45)
    # plt.ylabel('$\mathbf[m]$',  fontsize=45)
    # ax.legend(['x', 'y', 'z'], prop=dict(size='45'))
    # ax.set_title('$\mathbf{Right} \hspace{1cm} \mathbf{foot} \hspace{1cm} \mathbf{position}$', fontweight='bold', fontsize=55)
    # plt.show()

#################### JUMP TWIST ####################################
    with np.load('data_spin_z90.npz') as ref_data:
        copl = ref_data['copl']
        copr = ref_data['copr']
        time_sample = ref_data['time']
        com =  ref_data['com']
        com_vel = ref_data['dcom']
        com_acc = ref_data['ddcom']
        phase = ref_data['phase']
        ang = ref_data['ang']
        dang = ref_data['dang']
        q = ref_data['q']
        dq = ref_data['dq']

    with np.load('data_ang.npz') as ref_data:
        ang_real = ref_data['ang']
        quat_real = ref_data['quat']

    with np.load('data_lslide.npz') as ref_data:
        lslide = ref_data['lslide']
    with np.load('data_rslide.npz') as ref_data:
        rslide = ref_data['rslide']

    print('mean is ', np.mean(abs(lslide)))
    print('max is ', np.max(abs(lslide)))

    ang_real = ang_real[:,:1727]
    # ang_real[2,:] *= 0.8
    ang[2,:] *= 1.25
    quat_real = quat_real[:,:1727]
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    start = float((np.where(phase > 0)[0][0]+7)/1000)
    end = float((np.where(phase > 0)[0][-1]+7)/1000)
    len = q.shape[1]
    quater = np.zeros((4, len))
    for i in range(len):
        quater[:,i] = tf.quaternion_from_euler(q[0,i],q[1,i],q[2,i])

    # fig, ax = plt.subplots()
    # ax.plot(time_sample,lslide,linewidth=5)
    # ax.plot(time_sample,rslide,linewidth=5)
    # plt.axvline(x=start, color='r', linestyle='--', linewidth=5)
    # plt.axvline(x=end, color='r', linestyle='--', linewidth=5)
    # plt.xlabel('t [s]', fontsize=45)
    # plt.ylabel('[N.m]', fontsize=45)
    # ax.legend(['Left Slide', 'Right Slide'], prop=dict(size='45'))
    # ax.set_title('SLIDE Torque', fontweight='bold', fontsize=55)
    # plt.show()
#
    fig, ax = plt.subplots()
    ax.plot(time_sample, quater[0,:], linewidth=5)
    ax.plot(time_sample, quater[1,:], linewidth=5)
    ax.plot(time_sample, quater[2,:], linewidth=5)
    ax.plot(time_sample, quater[3,:], linewidth=5)
    ax.plot(time_sample, quat_real[0,:],  linestyle='-.',linewidth=5)
    ax.plot(time_sample, quat_real[1,:],  linestyle='-.',linewidth=5)
    ax.plot(time_sample, quat_real[2,:],  linestyle='-.',linewidth=5)
    ax.plot(time_sample, quat_real[3,:],  linestyle='-.',linewidth=5)
    plt.axvline(x=start, color='r', linestyle='--', linewidth=5)
    plt.axvline(x=end, color='r', linestyle='--', linewidth=5)
    plt.xlabel('$\mathbf{t} \hspace{0.5cm} \mathbf{[s]}$', fontsize=45)
    ax.legend(['$q_i$', '$q_j$', '$q_k$', '$q_w$'], prop=dict(size='45'))
    # plt.xlabel('',  fontsize=12)
    ax.set_title('$\mathbf{Base} \hspace{1cm} \mathbf{orientation}$', fontweight='bold', fontsize=55)
    plt.show()
#
    fig, ax = plt.subplots()
    ax.plot(time_sample, ang[0,:], linewidth=5)
    ax.plot(time_sample, ang[1,:], linewidth=5)
    ax.plot(time_sample, ang[2,:], linewidth=5)
    ax.plot(time_sample, ang_real[0,:],  linestyle='-.',linewidth=5)
    ax.plot(time_sample, ang_real[1,:],  linestyle='-.',linewidth=5)
    ax.plot(time_sample, ang_real[2,:],  linestyle='-.',linewidth=5)
    plt.axvline(x=start, color='r', linestyle='--', linewidth=5)
    plt.axvline(x=end+0.007, color='r', linestyle='--', linewidth=5)
    plt.xlabel('$\mathbf{t} \hspace{0.5cm} \mathbf{[s]}$', fontsize=45)
    plt.ylabel('$\mathbf{[kg \cdot m^{2}/s]}$',  fontsize=45)
    ax.legend(['$L_x$', '$L_y$', '$L_z$'], prop=dict(size='40'))
    ax.set_title('$\mathbf{Angular} \hspace{0.5cm} \mathbf{momentum}$', fontweight='bold', fontsize=55)
    plt.show()