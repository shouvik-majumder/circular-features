"""Rotate the circle and test whether the model's answer rotates.

If weekdays sit on a circle spanned by two directions u, v, and the model uses that circle to
compute "k days after X", then rotating a day's representation inside the (u, v) plane by k
steps should make the model answer as if the day were k later. The component outside the
plane is left unchanged.

A probe shows that the information is present; the rotation tests whether the model reads it
from there.

Controls:
  pc_plane       rotate by the same angle in the PC3-PC4 or PC5-PC6 plane of the same item
                 vectors, which displaces the activation by a comparable amount
  random_plane   rotate in a random 2D plane (barely moves the activation; reported for
                 completeness)
  displacement   how far the activation moved, so conditions can be compared
"""
from __future__ import annotations

import numpy as np
import torch


def circle_basis(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal basis (u, v) of the plane the items' circle lives in, plus the centre.

    X: [n_items, d_model] item vectors, in cyclic order. Returns (u, v, centre) with u, v
    orthonormal in the model's activation space.
    """
    mu = X.mean(0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    u, v = Vt[0], Vt[1]
    u = u / np.linalg.norm(u)
    v = v - (v @ u) * u
    v = v / np.linalg.norm(v)
    return u, v, mu


def item_angles(X: np.ndarray, u: np.ndarray, v: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Angle of each item around the circle, in the (u, v) plane."""
    Xc = X - centre
    return np.arctan2(Xc @ v, Xc @ u)


def rotation_matrix(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """Full-space matrix that rotates by `theta` inside span(u, v) and is the identity elsewhere.

    R = I + (cos t - 1)(uu^T + vv^T) + sin t (v u^T - u v^T)
    """
    d = len(u)
    uu = np.outer(u, u)
    vv = np.outer(v, v)
    return (np.eye(d) + (np.cos(theta) - 1.0) * (uu + vv)
            + np.sin(theta) * (np.outer(v, u) - np.outer(u, v)))


def random_plane(d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A random orthonormal 2D plane in the full space.

    The item vectors have almost no component in a random plane, so rotating there barely moves
    them. Reported for completeness; `pc_plane` is the primary control.
    """
    A = rng.standard_normal((d, 2))
    Q, _ = np.linalg.qr(A)
    return Q[:, 0], Q[:, 1]


def pc_plane(X: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """The plane spanned by principal components i and j of the item vectors.

    The primary control. PC3-PC4 carry real variance of the same data, so rotating there
    displaces the activations by a comparable amount, but they are not the circle. If the effect
    were simply 'large perturbations change the answer', this control would reproduce it.
    """
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    a, b = Vt[i], Vt[j]
    a = a / np.linalg.norm(a)
    b = b - (b @ a) * a
    return a, b / np.linalg.norm(b)


class RotationHook:
    """Applies a fixed rotation matrix to selected token positions of one layer's output."""

    def __init__(self, model, layer: int, hook: str = "resid_post"):
        self.name = f"blocks.{layer}.hook_{hook}"
        self.model = model
        self.R: torch.Tensor | None = None
        self.positions: list[int] | None = None
        self.centre: torch.Tensor | None = None

    def set(self, R: np.ndarray, positions: list[int], centre: np.ndarray, device, dtype) -> None:
        self.R = torch.as_tensor(R, device=device, dtype=dtype)
        self.centre = torch.as_tensor(centre, device=device, dtype=dtype)
        self.positions = positions

    def __call__(self, acts, hook):  # TransformerLens hook signature
        if self.R is None:
            return acts
        for p in self.positions:
            x = acts[:, p, :] - self.centre
            acts[:, p, :] = x @ self.R.T + self.centre
        return acts

    def displacement(self, X: np.ndarray, R: np.ndarray, centre: np.ndarray) -> float:
        """Mean L2 distance an item vector moves under this rotation, for norm matching."""
        Xc = X - centre
        return float(np.mean(np.linalg.norm(Xc @ R.T - Xc, axis=1)))
