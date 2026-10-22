"""A diffractometer as data: rotation axes, how they stack, and the beam.

A geometry is two stacks of circles, each listed outermost (closest to the
floor) first:

- the **sample** stack, whose product R = R_1 R_2 ... R_n turns the sample
  (phi) frame into the lab frame;
- the **detector** stack, whose product D turns the incoming beam direction
  into the outgoing one, k' = D k_in.

Each circle is a right-handed rotation about a fixed lab axis given at all
angles zero. That is the whole model; everything else (the forward map
angles -> hkl and the generated inverse solvers) follows from it.

Two ways to write one down:

- YAML (``Geometry.from_yaml``), or a dict with the same keys;
- xrayutilities' axis strings (``Geometry.from_xrayutilities``), e.g.
  ``sample=['x+', 'z-', 'y+', 'z-'], detector=['x+', 'z-'], r_i=(0, 1, 0)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from .rotation import rot, unit

# hc in eV * angstrom, same value (CODATA 2018) as scipy.constants
HC_EV_ANGSTROM = 12398.419843320026


@dataclass(frozen=True)
class Circle:
    name: str
    axis: tuple[float, float, float]      # unit vector, lab frame, all angles zero

    def matrix(self, angle_deg: float) -> np.ndarray:
        return rot(self.axis, np.radians(angle_deg))


@dataclass
class Geometry:
    name: str
    sample: list[Circle]
    detector: list[Circle]
    beam: tuple[float, float, float] = (0.0, 1.0, 0.0)
    description: str = ""
    labels: dict = field(default_factory=dict)   # e.g. {"x": "vertical"} for docs
    azimuth_zero: tuple | None = None   # lab direction where qaz / naz = 0 (default: see azimuth_frame)
    bisect_circle: str | None = None    # the circle whose axis 'bisect' / 'omega' refer to (psic: chi)

    def __post_init__(self):
        self.beam = tuple(unit(self.beam))
        names = [c.name for c in self.sample + self.detector]
        if len(set(names)) != len(names):
            raise ValueError(f"circle names must be unique, got {names}")
        if self.bisect_circle is not None and self.bisect_circle not in self.sample_names:
            raise ValueError(f"bisect_circle {self.bisect_circle!r} is not a sample circle")
        if self.azimuth_zero is not None:
            r = np.asarray(_axis(self.azimuth_zero), float)
            b = np.asarray(self.beam)
            r = r - b * (b @ r)
            self.azimuth_zero = tuple(unit(r))

    # --- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        def circles(items):
            return [Circle(c["name"], tuple(unit(_axis(c["axis"])))) for c in items]
        return cls(
            name=d["name"],
            sample=circles(d["sample"]),
            detector=circles(d["detector"]),
            beam=tuple(_axis(d.get("beam", (0, 1, 0)))),
            description=d.get("description", ""),
            labels=d.get("labels", {}),
            azimuth_zero=d.get("azimuth_zero"),
            bisect_circle=d.get("bisect_circle"),
        )

    @classmethod
    def from_yaml(cls, path_or_text) -> "Geometry":
        text = Path(path_or_text).read_text() if _is_path(path_or_text) else path_or_text
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_xrayutilities(
        cls,
        sample: list[str],
        detector: list[str],
        r_i=(0, 1, 0),
        sample_names: list[str] | None = None,
        detector_names: list[str] | None = None,
        name: str = "custom",
    ) -> "Geometry":
        """Build from xrayutilities ``QConversion`` axis strings.

        ``'x+'`` is a right-handed rotation about +x, ``'z-'`` about -z, as in
        xrayutilities; circles are listed outermost first, also as there.
        ``r_i`` is the primary-beam direction.
        """
        sn = sample_names or [f"s{i + 1}" for i in range(len(sample))]
        dn = detector_names or [f"d{i + 1}" for i in range(len(detector))]
        if len(sn) != len(sample) or len(dn) != len(detector):
            raise ValueError("one name per circle")
        return cls(
            name=name,
            sample=[Circle(n, tuple(_axis(a))) for n, a in zip(sn, sample)],
            detector=[Circle(n, tuple(_axis(a))) for n, a in zip(dn, detector)],
            beam=tuple(unit(r_i)),
        )

    @classmethod
    def builtin(cls, name: str) -> "Geometry":
        return cls.from_yaml(resources.files("hklforge.geometries").joinpath(f"{name}.yml").read_text())

    # --- introspection ------------------------------------------------------

    @property
    def sample_names(self) -> list[str]:
        return [c.name for c in self.sample]

    @property
    def detector_names(self) -> list[str]:
        return [c.name for c in self.detector]

    @property
    def names(self) -> list[str]:
        return self.sample_names + self.detector_names

    def azimuth_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """(r, u): azimuths about the beam (qaz, naz) are atan2(v.u, v.r).

        r is ``azimuth_zero`` or, by default, the lab axis after the beam's in
        cyclic order (beam y -> r = z); u = beam x r (beam y -> u = x). For psic
        that is diffcalc's qaz = atan2(Q_x, Q_z): 90 deg = vertical scattering.
        """
        b = np.asarray(self.beam)
        if self.azimuth_zero is not None:
            r = np.asarray(self.azimuth_zero)
        else:
            i = int(np.argmax(np.abs(b)))
            r = np.eye(3)[(i + 1) % 3]
            r = unit(r - b * (b @ r))
        return r, np.cross(b, r)

    def circle(self, name: str) -> Circle:
        for c in self.sample + self.detector:
            if c.name == name:
                return c
        raise KeyError(name)

    def __repr__(self):
        s = ", ".join(f"{c.name}{_axis_str(c.axis)}" for c in self.sample)
        d = ", ".join(f"{c.name}{_axis_str(c.axis)}" for c in self.detector)
        return f"Geometry({self.name!r}: sample [{s}] | detector [{d}] | beam {_axis_str(self.beam)})"

    # --- forward model ------------------------------------------------------

    def sample_matrix(self, angles: dict) -> np.ndarray:
        """R: phi (sample) frame -> lab frame."""
        R = np.eye(3)
        for c in self.sample:
            R = R @ c.matrix(angles.get(c.name, 0.0))
        return R

    def detector_matrix(self, angles: dict) -> np.ndarray:
        """D: incoming beam direction -> outgoing beam direction."""
        D = np.eye(3)
        for c in self.detector:
            D = D @ c.matrix(angles.get(c.name, 0.0))
        return D

    def k_out(self, angles: dict) -> np.ndarray:
        return self.detector_matrix(angles) @ np.asarray(self.beam)

    def q_lab(self, angles: dict, k: float) -> np.ndarray:
        """Scattering vector Q = k' - k in the lab frame, |k| = ``k``."""
        b = np.asarray(self.beam)
        return k * (self.k_out(angles) - b)

    def q_phi(self, angles: dict, k: float) -> np.ndarray:
        """Q in the sample (phi) frame."""
        return self.sample_matrix(angles).T @ self.q_lab(angles, k)


def wavevector(energy_eV: float) -> float:
    """|k| = 2 pi / lambda in 1/angstrom (the 2 pi convention, as diffcalc/libhkl)."""
    return 2 * np.pi * energy_eV / HC_EV_ANGSTROM


def _is_path(x) -> bool:
    return isinstance(x, Path) or (isinstance(x, str) and "\n" not in x and x.endswith((".yml", ".yaml")))


def _axis(a) -> np.ndarray:
    """'x+', '-y', '+z' or a 3-vector -> unit vector."""
    if isinstance(a, str):
        s = a.strip().lower()
        letter = s.strip("+-")
        sign = -1.0 if "-" in s else 1.0
        if letter not in ("x", "y", "z") or len(letter) != 1:
            raise ValueError(f"axis {a!r}: use x/y/z with a sign, e.g. 'z-'")
        v = np.zeros(3)
        v["xyz".index(letter)] = sign
        return v
    return unit(a)


def _axis_str(v) -> str:
    v = np.asarray(v)
    i = int(np.argmax(np.abs(v)))
    if np.isclose(abs(v[i]), 1.0):
        return f"({'xyz'[i]}{'+' if v[i] > 0 else '-'})"
    return f"({v[0]:.3g},{v[1]:.3g},{v[2]:.3g})"
