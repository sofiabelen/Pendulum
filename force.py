import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

## Reading parameters from file
param = np.genfromtxt('parameters.csv', delimiter=' ', names=True)

## Initialization
steps = int(param['steps'])
steps2 = int(param['steps2'])
omega0 = param['omega0']
phi0 = param['phi0']
# gamma = param['gamma']
v0 = param['v0']
f0 = param['f0']
sq_omega0 = omega0**2
dt = 0.01
omega_min = 0.1
omega_max = 3

## Force function
def f(t, omega):
    return f0 * np.cos(omega * t)

## Acceleration function
def a(t, v, phi, omega, gamma):
    return f(omega, t) - 2.0 * gamma * v - sq_omega0 * phi

## Defining simulate function
def simulate(omega, gamma):
    vn = v0
    phin = phi0

    # Af = np.zeros(steps)
    Afmax = 0

    for i in range(0, steps):
        an = a(i * dt, vn, phin, omega, gamma)
        v_prime = vn + an * dt
        phi = phin + vn * dt + an * dt**2 / 2.0
        phi_prime = phi
        a_prime = a(i * dt, v_prime, phi_prime, omega, gamma)
        v = vn + (an + a_prime) / 2.0 * dt

        phin = phi
        vn = v

        # Af[i] = np.sqrt(phi**2 + v**2)
        Afmax = max(Afmax, np.sqrt(phi**2 + v**2))

    return Afmax

def run(gamma):
    Afmax = np.zeros(steps2)
    omegas = np.zeros(steps2)
    j_char = 0
    
    for j in range(0, steps2):
        omega = omega_min + (omega_max - omega_min) * j / steps2
        # Af = simulate(omega, gamma)
        Afmax[j] = simulate(omega, gamma)
        omegas[j] = omega
    
        if(Afmax[j] > Afmax[j_char]):
            # Afmax[j_char] = Afmax[j]
            j_char = j
    
    if(gamma == 0.01):
        ## Asymptotes
        def line(x, a, b):
            return a*x + b
    
        j1 = int(steps2 / 10.0)
        # j2 = int(steps2 * 4.0 / 5.0)

        popt, pcov = curve_fit(f=line, xdata=omegas[:j1], ydata=Afmax[:j1])
        # popt2, pcov2 = curve_fit(f=line, xdata=omegas[j2:], ydata=Afmax[j2:])
        x = np.array([0, omegas[j1]])
        y = line(x, *popt)
        # x2 = np.array([omegas[j2], omegas[steps2 - 1] * 1.2])
        # y2 = line(x2, *popt2)
        # ax.plot(x, y, color='r', alpha=0.5)
        # ax.plot(x2, y2, color='r', alpha=0.5)
        ax.annotate(r'$A_f \to$ %.3f' % line(0, *popt), (0, 0))
        ax.annotate(r'$A_f \to$ 0', (2.7, -5))

    ## Plotting
    ax.plot(omegas, Afmax, label=r'$\gamma=$ %.2f' % gamma)
    ax.set_title(r'$A_f(\omega)$ при $\omega_0=$ %.1f' % omega0)

    # sns.regplot(x=omegas[:j1], y=Afmax[:j1], scatter=False, color='r')
    # sns.regplot(x=omegas[j2:], y=Afmax[j2:], scatter=False, color='g')

    # Maximum
    ax.axvline(omegas[j_char], alpha=0.5, color='black')
    ax.axhline(Afmax[j_char], alpha=0.5, color='black')
    ax.annotate(r'(%.2f, %.2f)'%(omegas[j_char], Afmax[j_char]),\
            (omegas[j_char] + 0.02, Afmax[j_char] + 0.001))



sns.set(context='notebook', palette='colorblind')
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$A_f$')

run(0.01)
run(0.1)
run(0.3)

ax.legend()
fig.savefig("Images/graph.pdf")
