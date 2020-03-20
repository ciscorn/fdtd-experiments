import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


IMP_0 = 376.730313668
C_0 = 299_792_458


class Grid:
    def __init__(self, size_x=200, c0dtds=1.0):
        self.q = 0
        self.size_x = size_x
        self.c0dtds = c0dtds  # courant number == c0 * (dt / dx)
        self.hy = np.zeros(size_x)
        self.ez = np.zeros(size_x)
        self.eps_r = np.ones(size_x)
        self.mu_r = np.ones(size_x)
        self.ceze = np.ones(size_x)
        self.cezh = np.ones(size_x)
        self.chyh = np.ones(size_x)
        self.chye = np.ones(size_x)
        self.setup()

    def calc_energy(self):
        a = np.sum(C_0 * self.eps_r / IMP_0 * (self.ez ** 2))
        b = np.sum((C_0 * self.mu_r * IMP_0 * (self.hy ** 2))[:-1])
        return (a + b) / 2

    def step(self):
        # update H
        ez = self.ez
        self.hy *= self.chyh
        self.hy += self.chye * np.diff(ez, append=0)

        # source
        self.source()

        # update E
        prev_ez1 = self.ez[1]
        hy = self.hy
        self.ez *= self.ceze
        self.ez += self.cezh * np.diff(hy, prepend=0)
        self.ez[0] = prev_ez1  # simple ABC
        self.ez[-1] = 0

        self.q += 1


class MyGrid(Grid):
    def setup(self):
        # media
        self.eps_r[100:] *= 4.0
        self.cezh *= self.c0dtds * IMP_0 / self.eps_r
        self.chye *= self.c0dtds / IMP_0 / self.mu_r
        loss_e = 0.0253146
        loss_h = 0
        self.cezh[100:] *= 1 / (1 + loss_e)
        self.ceze[100:] *= (1 - loss_e) / (1 + loss_e)
        self.chye[100:] *= 1 / (1 + loss_h)
        self.chyh[100:] *= (1 - loss_h) / (1 + loss_h)

    def _gaussian_source(self) -> None:
        pos = 50
        width = 10
        delay = 40

        imp = IMP_0 * np.sqrt(self.mu_r[pos - 1] / self.eps_r[pos])
        c1dtds = self.c0dtds / np.sqrt(self.mu_r[pos - 1] * self.eps_r[pos])
        q = self.q

        self.hy[pos - 1] -= np.exp(-(((c1dtds * (q - delay - 0.5) - 0.5) / width) ** 2)) / imp
        self.ez[pos] += np.exp(-((c1dtds * (q - delay) / width) ** 2)) / self.ceze[pos]

    def _harmonic_source(self) -> None:
        pos = 50
        n_lambda = 40  # wave length

        imp1 = IMP_0 * np.sqrt(self.mu_r[pos - 1] / self.eps_r[pos])
        c1dtds = self.c0dtds / np.sqrt(self.mu_r[pos - 1] * self.eps_r[pos])
        q = self.q

        self.hy[pos - 1] -= np.sin(2 * np.pi / n_lambda * (c1dtds * (q - 0.5) - 0.5)) / imp1
        self.ez[pos] += np.sin(2 * np.pi / n_lambda * c1dtds * q) / self.ceze[pos]

    def _richer_wavelet_source(self) -> None:
        pos = 50
        n_p = 40  # wave length of peak frequency
        delay = 1.3  # delay multiple

        imp1 = IMP_0 * np.sqrt(self.mu_r[pos - 1] / self.eps_r[pos])
        c1dtds = self.c0dtds / np.sqrt(self.mu_r[pos - 1] * self.eps_r[pos])
        q = self.q

        a_sq_h = (np.pi * ((c1dtds * (q - 0.5) - 0.5) / n_p - delay)) ** 2
        self.hy[pos - 1] -= (1 - 2 * a_sq_h) * np.exp(-a_sq_h) / imp1
        a_sq_e = (np.pi * ((c1dtds * q) / n_p - delay)) ** 2
        self.ez[pos] += (1 - 2 * a_sq_e) * np.exp(-a_sq_e) / self.ceze[pos]

    def source(self) -> None:
        #self._richer_wavelet_source()
        self._harmonic_source()


if __name__ == "__main__":
    grid = MyGrid(c0dtds=1.0)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim([-1.5, 1.5])
    ax2.set_ylim([-1 / 200, 1 / 200])
    ax1.set_ylabel("Ez [V/m]")
    ax2.set_ylabel("Hy [A/m]")
    x = np.arange(grid.size_x)
    emax = np.array(grid.ez)
    (lineEmax,) = ax1.plot(x, emax, color="tab:green")
    (lineE,) = ax1.plot(x, grid.ez, color="tab:blue")
    (lineH,) = ax2.plot(x[:-1] + 0.5, grid.hy[:-1], color="tab:orange")
    fig.tight_layout()

    def update_artists(frames):
        if frames > 100:
            np.maximum(grid.ez, emax, out=emax)
        lineEmax.set_ydata(emax)
        lineE.set_ydata(grid.ez)
        lineH.set_ydata(grid.hy[:-1])
        return (lineE, lineH, lineEmax)

    def step():
        for _ in range(500):
            if grid.q % 1 == 0:
                print("{0} {1:.4f}".format(grid.q, grid.calc_energy()))
                yield grid.q
            grid.step()

    FuncAnimation(fig, func=update_artists, frames=step, interval=100, blit=True)
    plt.show()
