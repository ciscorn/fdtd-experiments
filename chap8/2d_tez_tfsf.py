import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


IMP_0 = 376.730313668
C_0 = 299_792_458


class GridTEz:
    def __init__(self, size_x=200, size_y=160, c0dtds=1 / np.sqrt(2.0)):
        self.q = 0
        self.size_x = size_x
        self.size_y = size_y
        self.c0dtds = c0dtds  # courant number == c * (dt / dx)
        shape = (size_x, size_y)

        self.mu_r = np.ones(shape)
        self.eps_r_x = np.ones(shape)
        self.eps_r_y = np.ones(shape)

        self.ex = np.zeros(shape)
        self.ey = np.zeros(shape)
        self.hz = np.zeros(shape)

        self.cexh = np.ones(shape)
        self.cexe = np.ones(shape)
        self.ceyh = np.ones(shape)
        self.ceye = np.ones(shape)
        self.chze = np.ones(shape)
        self.chzh = np.ones(shape)

        self.setup()

        self._abc = ABC2nd()
        self._abc.initialize(self)

        self.aux_ey = np.zeros(size_x)
        self.aux_hz = np.zeros(size_x)
        ix = np.arange(size_x)
        loss_e = np.maximum(0, ((ix - size_x + 10) / 12) ** 3)
        loss_h = np.maximum(0, ((ix + 0.5 - size_x + 10) / 12) ** 3)
        self.aux_ceyh = self.ceyh[:, 100] * 1 / (1 + loss_e)
        self.aux_ceye = self.ceye[:, 100] * (1 - loss_e) / (1 + loss_e)
        self.aux_chze = self.chze[:, 100] * 1 / (1 + loss_h)
        self.aux_chzh = self.chzh[:, 100] * (1 - loss_h) / (1 + loss_h)

        # PEC
        iy, ix = np.meshgrid(np.arange(size_y), np.arange(size_x))
        mask1 = (ix + 0.5 - 100) ** 2 + (iy + 0.5 - 80) ** 2 <= 900
        mask2 = (ix - 0.5 - 100) ** 2 + (iy + 0.5 - 80) ** 2 <= 900
        mask3 = (ix + 0.5 - 100) ** 2 + (iy - 0.5 - 80) ** 2 <= 900
        self.ceye[mask1] = self.ceyh[mask1] = 0
        self.cexe[mask1] = self.cexh[mask1] = 0
        self.ceye[mask2] = self.ceyh[mask2] = 0
        self.cexe[mask3] = self.cexh[mask3] = 0

    def step(self):
        # update H
        self.hz *= self.chzh
        self.hz += self.chze * (
            np.diff(self.ex, axis=1, append=0) - np.diff(self.ey, axis=0, append=0)
        )

        # TFSF
        self.hz[9, 10:-11] += self.chze[9, 10:-11] * self.aux_ey[10]
        self.hz[-11, 10:-11] -= self.chze[-11, 10:-11] * self.aux_ey[-11]

        self.aux_hz *= self.aux_chzh
        self.aux_hz -= self.aux_chze * np.diff(self.aux_ey, append=0)
        self.aux_ey *= self.aux_ceye
        self.aux_ey -= self.aux_ceyh * np.diff(self.aux_hz, prepend=0)

        self.source2()

        self.ex[10:-11, 10] -= self.cexh[10:-11, 10] * self.aux_hz[10:-11]
        self.ex[10:-11, -11] += self.cexh[10:-11, -11] * self.aux_hz[10:-11]
        self.ey[10, 10:-11] += self.ceyh[10, 10:-11] * self.aux_hz[9]
        self.ey[-11, 10:-11] -= self.ceyh[-11, 10:-11] * self.aux_hz[-11]

        # update E
        self.ex *= self.cexe
        self.ex += self.cexh * np.diff(self.hz, axis=1, prepend=0)
        self.ey *= self.ceye
        self.ey -= self.ceyh * np.diff(self.hz, axis=0, prepend=0)

        # ABC
        self._abc.step(grid)

        self.q += 1


class MyGrid(GridTEz):
    def setup(self):
        self.cexh *= self.c0dtds * IMP_0 / self.eps_r_y
        self.ceyh *= self.c0dtds * IMP_0 / self.eps_r_x
        self.chze *= self.c0dtds / IMP_0 / self.mu_r

    def source2(self) -> None:
        n_p = 60  # wave length of peak frequency
        delay = 1.2  # delay multiple

        a_sq_e = (np.pi * ((self.c0dtds * self.q) / n_p - delay)) ** 2
        self.aux_ey[0] = (1 - 2 * a_sq_e) * np.exp(-a_sq_e) * 2 * IMP_0


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

        self._coefs_left = calc_coefs(grid.ceyh[0, :] * grid.chze[0, :])
        self._coefs_right = calc_coefs(grid.ceyh[-1, :] * grid.chze[-2, :])
        self._coefs_bottom = calc_coefs(grid.cexh[:, 0] * grid.chze[:, 0])
        self._coefs_top = calc_coefs(grid.cexh[:, -1] * grid.chze[:, -2])

        self._left = np.zeros((9, grid.size_y))
        self._left[3:6, :] = self._left[6:9, :] = grid.ey[0:3, :]
        self._right = np.zeros((9, grid.size_y))
        self._right[3:6, :] = self._right[6:9, :] = grid.ey[-1:-4:-1, :]
        self._bottom = np.zeros((9, grid.size_x))
        self._bottom[3:6, :] = self._bottom[6:9, :] = grid.ex[:, 0:3].T
        self._top = np.zeros((9, grid.size_x))
        self._top[3:6, :] = self._top[6:9, :] = grid.ex[:, -1:-4:-1].T

    def step(self, grid):
        def abc_edge(values, coefs, curr):
            values[0:3, :] = curr
            r0 = np.sum(coefs * values, axis=0)
            values[4:9, :] = values[1:6, :]
            values[3, :] = r0
            return r0

        grid.ey[0, :] = abc_edge(self._left, self._coefs_left, grid.ey[0:3, :])
        grid.ey[-1, :] = abc_edge(self._right, self._coefs_right, grid.ey[-1:-4:-1, :])
        grid.ex[:, 0] = abc_edge(self._bottom, self._coefs_bottom, grid.ex[:, 0:3].T).T
        grid.ex[:, -1] = abc_edge(self._top, self._coefs_top, grid.ex[:, -1:-4:-1].T).T


if __name__ == "__main__":
    grid = MyGrid()

    (x, y) = np.meshgrid(np.arange(grid.size_y), np.arange(grid.size_x))

    fig, ax1 = plt.subplots()
    ax1.set_aspect("equal")
    cmesh = ax1.pcolormesh(
        y, x, np.zeros(x.shape), cmap="twilight", vmin=-1.0, vmax=1.0, shading="flat",
    )
    fig.tight_layout()

    def update_artists(frames):
        cmesh.set_array(grid.hz[:-1, :-1].ravel())
        return (cmesh,)

    def step():
        for _ in range(2000):
            if grid.q % 2 == 0:
                print("{0}".format(grid.q))
                yield grid.q
            grid.step()

    FuncAnimation(fig, func=update_artists, frames=step, interval=50, blit=True)
    plt.show()
