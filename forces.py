import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

f0 = 5.0

def f2(t, omega):
    force = f0 * np.cos(omega * t)
    if np.cos(omega * t) < 0:
        force = 0
    return force

def f(t, omega):
    force = f0 * (1.0 / np.pi + 0.5 * np.cos(omega * t)\
            + 2.0 / (3.0 * np.pi) * np.cos(2.0 * omega * t)\
            - 2.0 / (15.0 * np.pi) * np.cos(4.0 * omega * t))
    return force

sns.set(context='notebook', palette='colorblind')
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$f(t)$')
ax.set_title("Forces")

time = np.arange(0, 100.0, 0.01)
omega = 0.5
ax.plot(time, f(time, omega), label=r'$\omega=$%.1f : Fourier' % omega)
omega = 0.1
ax.plot(time, f(time, omega), label=r'$\omega=$%.1f : Fourier' % omega)

force = np.arange(0, 100.0, 0.01)
omega = 0.5
for i in range(len(time)):
    force[i] = f2(time[i], omega)
ax.plot(time, force, label=r'$\omega=$%.1f' % omega)
omega = 0.1
for i in range(len(time)):
    force[i] = f2(time[i], omega)
ax.plot(time, force, label=r'$\omega=$%.1f' % omega)

ax.legend()
fig.savefig("img/forces.pdf")
fig.savefig("img/forces.svg")
