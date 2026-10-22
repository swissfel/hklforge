"""Crystal lattice -> B matrix (Busing & Levy 1967), 2 pi convention.

``ub = u @ b_matrix(a, b, c, alpha, beta, gamma)`` gives q = UB h with
|q| = 2 pi / d, the convention of the rest of hklforge (and of diffcalc and
libhkl). Lengths in angstrom, angles in degrees.
"""
from __future__ import annotations

import numpy as np


def b_matrix(a: float, b: float, c: float, alpha: float = 90.0, beta: float = 90.0,
             gamma: float = 90.0) -> np.ndarray:
    al, be, ga = np.radians([alpha, beta, gamma])
    # real-space metric -> reciprocal lattice (with 2 pi)
    A = np.array([[a, b * np.cos(ga), c * np.cos(be)],
                  [0, b * np.sin(ga), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
                  [0, 0, 0]])
    A[2, 2] = np.sqrt(c * c - A[0, 2] ** 2 - A[1, 2] ** 2)
    rec = 2 * np.pi * np.linalg.inv(A).T                 # columns a*, b*, c*
    G = rec.T @ rec                                      # reciprocal metric
    ast, bst, cst = np.sqrt(np.diag(G))
    cos_al_st = G[1, 2] / (bst * cst)
    cos_be_st = G[0, 2] / (ast * cst)
    cos_ga_st = G[0, 1] / (ast * bst)
    sin_be_st, sin_ga_st = np.sqrt(1 - cos_be_st ** 2), np.sqrt(1 - cos_ga_st ** 2)
    return np.array([[ast, bst * cos_ga_st, cst * cos_be_st],
                     [0, bst * sin_ga_st, -cst * sin_be_st * np.cos(al)],
                     [0, 0, 2 * np.pi / c]])


def d_spacing(b: np.ndarray, hkl) -> float:
    return 2 * np.pi / np.linalg.norm(b @ np.asarray(hkl, float))
