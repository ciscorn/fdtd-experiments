import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


N = 200
IMP_0 = 376.730313668
C_0 = 299_792_458
courant = 1.0
assert courant == 1.0
dx = 1
dt = courant * dx / C_0  # courant == (dt / dx) * c

Hy = np.zeros(N)
Ez = np.zeros(N)
x = np.arange(0, N, dx)

eps_r = np.ones(Ez.shape)
mu_r = np.ones(Ez.shape)
eps_r[100:] = 9.0

loss_e = np.maximum(0, ((x - 190) / 12) ** 3)
loss_h = np.maximum(0, ((x + 0.5 - 190) / 12) ** 3)
ceze = (1 - loss_e) / (1 + loss_e)
cezh = (courant * IMP_0 / eps_r) / (1 + loss_e)
chyh = (1 - loss_h) / (1 + loss_h)
chye = (courant / IMP_0 / mu_r) / (1 + loss_h)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_ylim([-1.5, 1.5])
ax2.set_ylim([-1 / 200, 1 / 200])
ax1.set_ylabel("Ez [V/m]")
ax2.set_ylabel("Hy [A/m]")
(lineE,) = ax1.plot(x, Ez, color="tab:blue")
(lineH,) = ax2.plot(x[:-1] + 0.5, Hy[:-1], color="tab:orange")
fig.tight_layout()


# loss2 = (loss + np.roll(loss, -1)) / 2
# chyh = (1 - loss2) / (1 + loss2)
# chye = (courant / IMP_0 / mu_r) / (1 + loss2)


def step():
    global Hy, Ez

    for q in range(2000):
        a = np.sum(C_0 * eps_r / IMP_0 * (Ez ** 2))
        b = np.sum((C_0 * mu_r * IMP_0 * (Hy ** 2))[:-1])
        print("{0:.4f} {1:.4f} {2:.4f}".format(a, b, a + b))

        Hy = chyh * Hy + chye * (np.roll(Ez, -1) - Ez)

        Hy[49] -= np.exp(-(((q - 0.5 - 0.5 - 40) / 13) ** 2)) / IMP_0

        tmp = Ez[1]
        Ez = ceze * Ez + cezh * (Hy - np.roll(Hy, 1))
        Ez[0] = tmp  # simple ABC
        Ez[-1] = 0

        Ez[50] += np.exp(-(((q - 40) / 13) ** 2))

        if q % 5 == 0:
            yield


def update_artists(frames):
    lineE.set_ydata(Ez)
    lineH.set_ydata(Hy[:-1])
    return (lineE, lineH)


FuncAnimation(fig, func=update_artists, frames=step, interval=100, blit=True)
plt.show()
