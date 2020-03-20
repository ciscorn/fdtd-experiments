import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


N = 200
IMP_0 = 376.730313668
C_0 = 299792458
courant = 1.00
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
(lineH,) = ax2.plot(x, Hy, color="tab:orange")
fig.tight_layout()


def step():
    global Hy, Ez
    for q in range(2000):
        print(q, q * dt)

        tmp = courant * Hy[-2]
        Hy += courant / IMP_0 * (np.roll(Ez, -1) - Ez)
        Hy[-1] = tmp  # simple ABC

        tmp = courant * Ez[1]
        Ez += courant * IMP_0 * (Hy - np.roll(Hy, 1))
        Ez[0] = tmp  # simple ABC

        # additive source
        Ez[100] += np.exp(-(((q - 30) / 10) ** 2))

        yield


def update_artists(frames):
    lineE.set_ydata(Ez)
    lineH.set_ydata(Hy)
    return (lineE, lineH)


FuncAnimation(fig, func=update_artists, frames=step, interval=10, blit=True)
plt.show()
