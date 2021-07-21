import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
# Degree of interpolating polynomial
d = 4
g = 9.81
l = 1.0
m = 1.0
T = 1

# Get collocation points
# tau_root = np.append(0, ca.collocation_points(d, 'legendre'))
tau = ca.collocation_points(d, 'legendre')
# Collocation linear maps
[C,D,B] = ca.collocation_coeff(tau)

# # Coefficients of the collocation equation
# C = np.zeros((d+1,d+1))
#
# # Coefficients of the continuity equation
# D = np.zeros(d+1)
#
# # Coefficients of the quadrature function
# B = np.zeros(d+1)


# Declare model variables
x1 = ca.MX.sym('theta')
x2 = ca.MX.sym('theta_dot')
x = ca.vertcat(x1, x2)
u = ca.MX.sym('u')

# Model equations
xdot = ca.vertcat(x2, -g/l*ca.sin(x1)+u)

# Objective term
L = u**2

# Continuous time dynamics
f = ca.Function('f', [x, u], [xdot, L])

# Control discretization
N = 100 # number of control intervals
h = T/N

opti = ca.Opti()
J = 0

# "Lift" initial conditions
Xk = opti.variable(2, N+1)
opti.subject_to(Xk[:,0] == [0, 0])
opti.subject_to(Xk[:,-1] == [3.14, 0])

# opti.set_initial(Xk[:, 0], [0, 0])

Uk = opti.variable(1, N)
# opti.subject_to(Uk[0] == [0])
# opti.subject_to(Uk[-1] == [0])
# Collect all states/controls
# Xs = [Xk]
# Us = []

# Formulate the NLP
for k in range(0, N):
   # New NLP variable for the control
   #  Uk = opti.variable(1)
    # Us.append(Uk)
    # opti.subject_to(-30 <= Uk[k])
    # opti.subject_to(Uk[k] <= 30)
    # opti.set_initial(Uk[k], 0)

    # Decision variables for helper states at each collocation point
    Xc = opti.variable(2, d)
    # opti.subject_to(-0.25 <= Xc[1,:])
    opti.set_initial(Xc, np.zeros((2,d)))

    # Evaluate ODE right-hand-side at all helper states
    ode, quad = f(Xc, Uk[k])
    # Add contribution to quadrature function
    J += quad@B*h

    # Get interpolating points of collocation polynomial
    # Z = [Xk, Xc]
    Z = ca.horzcat(Xk[:,k], Xc)
    # Get slope of interpolating polynomial (normalized)
    Pidot = Z@C
    # Match with ODE right-hand-side
    opti.subject_to(Pidot == h@ode)

    # State at end of collocation interval
    Xk_end = Z@D

    # New decision variable for state at end of interval
    # Xk = opti.variable(2)
    # Xs.append(Xk)
    # opti.subject_to(-0.25 <= Xk[1, k+1])
    opti.set_initial(Xk[:, k+1], np.zeros(2))

    # Continuity constraints
    opti.subject_to(Xk_end == Xk[:, k+1])

# Xs = [Xs{:}];
# Us = [Us{:}];

opti.minimize(J)

opti.solver('ipopt')

sol = opti.solve()

x_opt = sol.value(Xk)
u_opt = sol.value(Uk)

print(np.sum(np.array(u_opt)**2) * T / N)
fig, axs = plt.subplots(2)
axs[0].plot(x_opt.T)
axs[0].legend(['x', 'x_dot'])
axs[1].plot(u_opt.T)
axs[1].legend(['u'])
plt.show()


