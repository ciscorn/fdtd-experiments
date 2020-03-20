import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


N = 200
IMP_0 = 376.730313668
C_0 = 299_792_458
courant = 1.0
assert courant == 1.0
dx = 1
dt = courant * dx / C_0  # (dt / dx) * c == courant

Hy = np.zeros(N)
Ez = np.zeros(N)
x = np.arange(0, N, dx)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_ylim([-1.5, 1.5])
ax2.set_ylim([-1 / 200, 1 / 200])
ax1.set_ylabel("Ez [V/m]")
ax2.set_ylabel("Hy [A/m]")
(lineE,) = ax1.plot(x, Ez, color="tab:blue")
(lineH,) = ax2.plot(x[:-1] + 0.5, Hy[:-1], color="tab:orange")
fig.tight_layout()


eps_r = np.ones(Ez.shape) * 1.0
eps_r[100:] = 9.0

mu_r = np.ones(Ez.shape) * 1.0
mu_r[100:] = 1.0


def step():
    global Hy, Ez

    for q in range(2000):
        a = np.sum(C_0 * eps_r / IMP_0 * (Ez ** 2))
        b = np.sum(C_0 * mu_r * IMP_0 * (Hy ** 2))
        print("{0:.4f} {1:.4f} {2:.4f}".format(a, b, a + b))

        Hy += courant / IMP_0 / mu_r * (np.roll(Ez, -1) - Ez)

        if q >= 1:
            Hy[49] -= np.exp(-(((q - 0.5 - 0.5 - 40) / 13) ** 2)) / IMP_0

        (tmp, tmp2) = (Ez[1], Ez[-2])
        Ez += courant * IMP_0 / eps_r * (Hy - np.roll(Hy, 1))
        (Ez[0], Ez[-1]) = (tmp, tmp2)  # simple ABC

        # additive source
        Ez[50] += np.exp(-(((q - 40) / 13) ** 2))

        yield


def update_artists(frames):
    lineE.set_ydata(Ez)
    lineH.set_ydata(Hy[:-1])
    return (lineE, lineH)


FuncAnimation(fig, func=update_artists, frames=step, interval=100, blit=True)
plt.show()
