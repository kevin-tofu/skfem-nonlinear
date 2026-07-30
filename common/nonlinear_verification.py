"""Numerical verification helpers for nonlinear residuals and tangents."""

import numpy as np


def directional_derivative_errors(residual, jacobian, state, free, direction=None, steps=None):
    """Compare J(u)d with a centred finite difference of R(u).

    Parameters
    ----------
    residual
        Callable returning the full residual vector.
    jacobian
        Callable returning the full tangent matrix.
    state
        State vector at which the check is performed.
    free
        Degrees of freedom included in the norm.
    direction
        Optional perturbation direction.  A reproducible random direction is
        generated on ``free`` when omitted.
    steps
        Finite-difference step sizes.
    """
    state = np.asarray(state)
    free = np.asarray(free)
    if direction is None:
        generator = np.random.default_rng(20260730)
        direction = np.zeros_like(state)
        direction[free] = generator.standard_normal(free.size)
    direction = np.asarray(direction)
    direction /= np.linalg.norm(direction[free])
    if steps is None:
        steps = np.logspace(-2, -7, 11)
    steps = np.asarray(steps)

    exact = (jacobian(state) @ direction)[free]
    scale = max(np.linalg.norm(exact), np.finfo(float).eps)
    errors = []
    for step in steps:
        difference = (
            residual(state + step * direction) - residual(state - step * direction)
        ) / (2.0 * step)
        errors.append(np.linalg.norm(difference[free] - exact) / scale)
    return steps, np.asarray(errors)
