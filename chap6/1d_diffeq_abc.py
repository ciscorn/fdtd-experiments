import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


IMP_0 = 376.730313668
C_0 = 299_792_458


class Grid:
    def __init__(self, size_x=200, c0dtds=1.0):
        self.q = 0
        self.size_x = size_x
        self.c0dtds = c0dtds  # courant number == c * (dt / dx)
        self.hy = np.zeros(size_x)
        self.ez = np.zeros(size_x)
        self.eps_r = np.ones(size_x)
        self.mu_r = np.ones(size_x)
        self.ceze = np.ones(size_x)
        self.cezh = np.ones(size_x)
        self.chyh = np.ones(size_x)
        self.chye = np.ones(size_x)
        self.setup()
        self._abc = ABC_2nd()
        self._abc.initialize(self)

    def calc_energy(self):
        a = np.sum(C_0 * self.eps_r / IMP_0 * (self.ez ** 2))
        b = np.sum((C_0 * self.mu_r * IMP_0 * (self.hy ** 2))[:-1])
        return (a + b) / 2

    def step(self):
        # update H
        self.hy *= self.chyh
        self.hy += self.chye * np.diff(self.ez, append=0)

        # source
        self.source()

        # update E
        self.ez *= self.ceze
        self.ez += self.cezh * np.diff(self.hy, prepend=0)

        # ABC
        self._abc.step(grid)

        self.q += 1


class ABC_1st:
    # First Order 1-D ABC
    def initialize(self, grid):
        sc1_left = np.sqrt(grid.cezh[0] * grid.chye[0])
        sc1_right = np.sqrt(grid.cezh[-1] * grid.chye[-2])
        self._coef_left = (sc1_left - 1) / (sc1_left + 1)
        self._coef_right = (sc1_right - 1) / (sc1_right + 1)
        self._prev_l0 = grid.ez[0]
        self._prev_l1 = grid.ez[1]
        self._prev_r0 = grid.ez[-1]
        self._prev_r1 = grid.ez[-2]

    def step(self, grid):
        grid.ez[0] = self._prev_l1 + self._coef_left * (grid.ez[1] - self._prev_l0)
        grid.ez[-1] = self._prev_r1 + self._coef_right * (grid.ez[-2] - self._prev_r0)
        self._prev_l0 = grid.ez[0]
        self._prev_l1 = grid.ez[1]
        self._prev_r0 = grid.ez[-1]
        self._prev_r1 = grid.ez[-2]


class ABC_2nd:
    # Second Order 1-D ABC
    def initialize(self, grid):
        def calc_coefs(cezh_x, chye_x):
            sc1 = np.sqrt(cezh_x * chye_x)
            sc1_inv = 1.0 / sc1
            co2 = sc1_inv + 2.0 + sc1
            return (
                -(sc1_inv - 2.0 + sc1) / co2,
                -2.0 * (sc1 - sc1_inv) / co2,
                4.0 * (sc1 + sc1_inv) / co2,
            )

        self._coefs_left = calc_coefs(grid.cezh[0], grid.chye[0])
        self._coefs_right = calc_coefs(grid.cezh[-1], grid.chye[-2])
        self._prev2_l = grid.ez[:3].copy()
        self._prev2_r = grid.ez[-3:].copy()
        self._prev1_l = grid.ez[:3].copy()
        self._prev1_r = grid.ez[-3:].copy()

    def step(self, grid):
        ez = grid.ez

        (cl0, cl1, cl2) = self._coefs_left
        prev2_l = self._prev2_l
        prev1_l = self._prev1_l
        grid.ez[0] = (
            cl0 * (ez[2] + prev2_l[0])
            + cl1 * (prev1_l[0] + prev1_l[2] - ez[1] - prev2_l[1])
            + cl2 * prev1_l[1]
            - prev2_l[2]
        )
        (cr0, cr1, cr2) = self._coefs_right
        prev2_r = self._prev2_r
        prev1_r = self._prev1_r
        ez[-1] = (
            cr0 * (ez[-3] + prev2_r[2])
            + cr1 * (prev1_r[2] + prev1_r[0] - ez[-2] - prev2_r[1])
            + cr2 * prev1_r[1]
            - prev2_r[0]
        )

        self._prev2_l = self._prev1_l
        self._prev2_r = self._prev1_r
        self._prev1_l = grid.ez[:3].copy()
        self._prev1_r = grid.ez[-3:].copy()


class MyGrid(Grid):
    def setup(self):
        # media
        self.eps_r[100:] *= 9.0
        self.cezh *= self.c0dtds * IMP_0 / self.eps_r
        self.chye *= self.c0dtds / IMP_0 / self.mu_r

    def source(self) -> None:
        # TFSF harmonic source
        pos = 50
        n_lambda = 40  # wave length

        imp1 = IMP_0 * np.sqrt(self.mu_r[pos - 1] / self.eps_r[pos])
        c1dtds = self.c0dtds / np.sqrt(self.mu_r[pos - 1] * self.eps_r[pos])
        q = self.q

        self.hy[pos - 1] -= (
            np.sin(2 * np.pi / n_lambda * (c1dtds * (q - 0.5) - 0.5)) / imp1
        )
        self.ez[pos] += np.sin(2 * np.pi / n_lambda * c1dtds * q) / self.ceze[pos]


if __name__ == "__main__":
    grid = MyGrid(c0dtds=1.0)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim([-1.5, 1.5])
    ax2.set_ylim([-1 / 200, 1 / 200])
    ax1.set_ylabel("Ez [V/m]")
    ax2.set_ylabel("Hy [A/m]")
    x = np.arange(grid.size_x)
    (lineE,) = ax1.plot(x, grid.ez, color="tab:blue")
    (lineH,) = ax2.plot(x[:-1] + 0.5, grid.hy[:-1], color="tab:orange")
    fig.tight_layout()

    def update_artists(frames):
        lineE.set_ydata(grid.ez)
        lineH.set_ydata(grid.hy[:-1])
        return (lineE, lineH)

    def step():
        for _ in range(2000):
            if grid.q % 5 == 0:
                print("{0} {1:.4f}".format(grid.q, grid.calc_energy()))
                yield
            grid.step()

    FuncAnimation(fig, func=update_artists, frames=step, interval=100, blit=True)
    plt.show()
