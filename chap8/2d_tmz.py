import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


IMP_0 = 376.730313668
C_0 = 299_792_458


class GridTMz:
    def __init__(self, size_x=320, size_y=200, c0dtds=1 / np.sqrt(2.0)):
        self.q = 0
        self.size_x = size_x
        self.size_y = size_y
        self.c0dtds = c0dtds  # courant number == c * (dt / dx)
        shape = (size_x, size_y)

        self.eps_r = np.ones(shape)
        self.mu_r_x = np.ones(shape)
        self.mu_r_y = np.ones(shape)

        self.hx = np.zeros(shape)
        self.hy = np.zeros(shape)
        self.ez = np.zeros(shape)

        self.chyh = np.ones(shape)
        self.chye = np.ones(shape)
        self.chxh = np.ones(shape)
        self.chxe = np.ones(shape)
        self.ceze = np.ones(shape)
        self.cezh = np.ones(shape)

        self.setup()

        self._abc = ABC2nd()
        self._abc.initialize(self)

    def step(self):
        # update H
        self.hx *= self.chxh
        self.hx -= self.chxe * np.diff(self.ez, axis=1, append=0)
        self.hy *= self.chyh
        self.hy += self.chye * np.diff(self.ez, axis=0, append=0)

        # source
        self.source()

        # update E
        self.ez *= self.ceze
        self.ez += self.cezh * (
            np.diff(self.hy, axis=0, prepend=0) - np.diff(self.hx, axis=1, prepend=0)
        )

        # ABC
        self._abc.step(grid)

        self.q += 1


class MyGrid(GridTMz):
    def setup(self):
        # media
        self.eps_r[50:200, 50:80] = 9
        self.eps_r[50:200, 120:150] = 9

        self.chye *= self.c0dtds / IMP_0 / self.mu_r_x
        self.chxe *= self.c0dtds / IMP_0 / self.mu_r_y
        self.cezh *= self.c0dtds * IMP_0 / self.eps_r

    def source(self) -> None:
        pos = (100, 100)
        n_p = 60  # wave length of peak frequency
        delay = 1.2  # delay multiple

        a_sq_e = (np.pi * ((self.c0dtds * self.q) / n_p - delay)) ** 2
        self.ez[pos] += (1 - 2 * a_sq_e) * np.exp(-a_sq_e) / self.ceze[pos] * 50


class ABC2nd:
    def initialize(self, grid):
        def calc_coefs(imps):
            sc1 = np.sqrt(imps)
            sc1_inv = 1.0 / sc1
            co2 = sc1_inv + 2.0 + sc1
            c0 = -(sc1_inv - 2.0 + sc1) / co2
            c1 = -2.0 * (sc1 - sc1_inv) / co2
            c2 = 4.0 * (sc1 + sc1_inv) / co2
            return np.vstack(
                [np.zeros(c1.shape), -c1, c0, c1, c2, c1, c0, -c1, -np.ones(c1.shape)]
            )

        self._coefs_left = calc_coefs(grid.cezh[0, :] * grid.chye[0, :])
        self._coefs_right = calc_coefs(grid.cezh[-1, :] * grid.chye[-2, :])
        self._coefs_bottom = calc_coefs(grid.cezh[:, 0] * grid.chxe[:, 0])
        self._coefs_top = calc_coefs(grid.cezh[:, -1] * grid.chxe[:, -2])

        self._left = np.zeros((9, grid.size_y))
        self._left[3:6, :] = self._left[6:9, :] = grid.ez[0:3, :]
        self._right = np.zeros((9, grid.size_y))
        self._right[3:6, :] = self._right[6:9, :] = grid.ez[-1:-4:-1, :]
        self._bottom = np.zeros((9, grid.size_x))
        self._bottom[3:6, :] = self._bottom[6:9, :] = grid.ez[:, 0:3].T
        self._top = np.zeros((9, grid.size_x))
        self._top[3:6, :] = self._top[6:9, :] = grid.ez[:, -1:-4:-1].T

    def step(self, grid):
        def abc_edge(values, coefs, curr):
            values[0:3, :] = curr
            r0 = np.sum(coefs * values, axis=0)
            values[4:9, :] = values[1:6, :]
            values[3, :] = r0
            return r0

        grid.ez[0, :] = abc_edge(self._left, self._coefs_left, grid.ez[0:3, :])
        grid.ez[-1, :] = abc_edge(self._right, self._coefs_right, grid.ez[-1:-4:-1, :])
        grid.ez[:, 0] = abc_edge(self._bottom, self._coefs_bottom, grid.ez[:, 0:3].T).T
        grid.ez[:, -1] = abc_edge(self._top, self._coefs_top, grid.ez[:, -1:-4:-1].T).T


if __name__ == "__main__":
    grid = MyGrid()

    (x, y) = np.meshgrid(np.arange(grid.size_y + 1), np.arange(grid.size_x + 1))

    fig, ax1 = plt.subplots()
    ax1.set_aspect("equal")
    cmesh = ax1.pcolormesh(
        y, x, np.zeros(x.shape), cmap="PiYG", vmin=-1.0, vmax=1.0, shading="flat"
    )
    fig.tight_layout()

    def update_artists(frames):
        cmesh.set_array(grid.ez.ravel())
        return (cmesh,)

    def step():
        for _ in range(2000):
            if grid.q % 5 == 0:
                print("{0}".format(grid.q))
                yield grid.q
            grid.step()

    FuncAnimation(fig, func=update_artists, frames=step, interval=50, blit=True)
    plt.show()
