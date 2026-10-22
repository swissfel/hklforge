"""Regenerate the figures in docs/figures (python docs/make_figures.py)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from hklforge import Diffractometer, Geometry  # noqa: E402
from hklforge.draw import draw, draw_circles  # noqa: E402

PSIC = Geometry.builtin("psic")
# a cubic crystal, a = 4 A, U = identity; surface normal (0, 0, 1), reference (0, -1, 1)
UB = 2 * np.pi / 4.0 * np.eye(3)
D = Diffractometer(PSIC, UB, energy=9000, surface_hkl=(0, 0, 1), reference_hkl=(0, -1, 1))
# reference vector = surface normal (0, 0, 1), for the incidence / exit figures
D_UP = Diffractometer(PSIC, UB, energy=9000, surface_hkl=(0, 0, 1), reference_hkl=(0, 0, 1))
# the branch whose naz is far from both 0 and qaz, so the azimuth figure reads well
POS = max(D.solve((1, 1, 2), {"mu": 10, "nu": 20, "betain": 5}),
          key=lambda a: min(abs(D.pseudo(a)["naz"]), abs(D.pseudo(a)["naz"] - D.pseudo(a)["qaz"])))

FIGURES = {
    "scattering": dict(show=("ttheta",), view="plane", title="b, k', Q and 2θ; the shaded plane is the scattering plane"),
    "azimuths": dict(show=("qaz", "naz"), view="beam",
                     title="qaz and naz: azimuths about the beam (looking downstream)"),
    "reference_angles": dict(show=("alpha", "beta"), mode={"mu": 0, "nu": 0, "alpha": 10},
                             d=D_UP, title="alpha, beta: beams to the plane normal to n"),
    "tau": dict(show=("tau",), view="Qn", title="tau: angle between Q and the reference vector n"),
    "surface_angles": dict(show=("betain", "betaout"), mode={"mu": 0, "nu": 0, "betain": 10},
                           d=D_UP, title="betain, betaout: beams to the surface"),
    "psi": dict(show=("psi",), view="Q", title="psi: azimuth of n about Q, from p (looking down Q)"),
    "omega": dict(show=("omega",), mode={"nu": 20, "bisect": True, "omega": 20},
                  title="bisect + omega: the chi axis in the scattering plane, omega from p"),
}


def main():
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    for name, kw in FIGURES.items():
        kw = dict(kw)
        mode, d = kw.pop("mode", None), kw.pop("d", D)
        pos = POS if mode is None else min(d.solve((1, 1, 2), mode),
                                           key=lambda a: sum(abs(v) for v in a.values()))
        ax = draw(d, pos, **kw)
        ax.figure.savefig(os.path.join(out, f"{name}.png"), dpi=110, bbox_inches="tight")
        plt.close(ax.figure)
    ax = draw_circles(PSIC, {"mu": 10, "eta": 20, "chi": 30, "phi": 0, "nu": 15, "delta": 30},
                      title="psic: mu, eta, chi, phi (sample), nu, delta (detector)")
    ax.figure.savefig(os.path.join(out, "psic_circles.png"), dpi=110, bbox_inches="tight")
    plt.close(ax.figure)
    print("position used:", {k: round(v, 2) for k, v in POS.items()})
    print("virtual angles:", {k: round(v, 2) for k, v in D.pseudo(POS).items()})


if __name__ == "__main__":
    main()
