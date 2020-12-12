import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

## Initialization
steps_omega = (int)(1000)
omega0 = 1.0
phi0 = 1.0
v0 = 1.0
f0 = 5.0
sq_omega0 = omega0**2
omega_min = 0.1
omega_max = 1.0
dt = 0.01

## Force function
def f(t, omega):
    force = f0 * np.cos(omega * t)
    # if np.cos(omega * t) < 0:
    #     force = 0

    ## Fourier
    # force = f0 * (1.0 / np.pi + 0.5 * np.cos(omega * t)\
    #         + 2.0 / (3.0 * np.pi) * np.cos(2.0 * omega * t)\
    #         - 2.0 / (15.0 * np.pi) * np.cos(4.0 * omega * t))

    return force

## Acceleration function
def a(t, v, phi, omega, gamma):
    return f(omega, t) - 2.0 * gamma * v - sq_omega0 * phi

## DE Solver: Predictor corrector method
def solver(omega, gamma, steps_sim):
    vn = v0
    phin = phi0

    Afmax = 0

    for i in range(0, steps_sim):
        an = a(i * dt, vn, phin, omega, gamma)
        v_prime = vn + an * dt
        phi = phin + vn * dt + an * dt**2 / 2.0
        phi_prime = phi
        a_prime = a(i * dt, v_prime, phi_prime, omega, gamma)
        v = vn + (an + a_prime) / 2.0 * dt

        phin = phi
        vn = v

        if(i >= 0.6 * float(steps_sim)):
            Afmax = max(Afmax, np.abs(phi))

    return Afmax

def run(gamma):
    Afmax = np.zeros(steps_omega)
    omegas = np.zeros(steps_omega)
    j_char = 0

    total_run_time = 3.0 / gamma
    steps_sim = int(total_run_time / dt)
    print(steps_sim)

    for j in range(0, steps_omega):
        omega = omega_min + (omega_max - omega_min) * j / steps_omega

        Afmax[j] = solver(omega, gamma, steps_sim)

        omegas[j] = omega
    
        if(Afmax[j] > Afmax[j_char]):
            j_char = j
    
    ## Asymptotes
    def line(x, a, b):
        return a*x + b
    
    j1 = int(steps_omega / 5.0)

    popt, pcov = curve_fit(f=line, xdata=omegas[:j1], ydata=Afmax[:j1])
    x = np.array([0, omegas[j1]])
    y = line(x, *popt)
    ax.annotate(r'$A_f \to$ 0', (1.7, 0))

    ## Plotting
    ax.scatter(omegas, Afmax, label=r'$\gamma=$ %.2f' % gamma, s=0.7)
    ax.set_title(r'$A_f(\omega)$')

    ## Maximum
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
fig.savefig("img/graph_1.pdf")
