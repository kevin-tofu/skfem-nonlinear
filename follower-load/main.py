"""Finite-deformation cantilever: dead load versus follower load."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, FacetBasis, LinearForm
from skfem import MeshTri, asm
from skfem.helpers import ddot, dot, grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.nonlinear_verification import directional_derivative_errors


MU, LAME_LAMBDA = 1.0, 2.0
LENGTH, HEIGHT = 2.0, 0.4


def constitutive(F):
    moved = np.moveaxis(F, (0, 1), (-2, -1))
    determinant = np.linalg.det(moved)
    inverse_transpose = np.swapaxes(np.linalg.inv(moved), -1, -2)
    inverse_transpose = np.moveaxis(inverse_transpose, (-2, -1), (0, 1))
    stress = (
        MU * (F - inverse_transpose)
        + LAME_LAMBDA * np.log(determinant)[None, None, ...] * inverse_transpose
    )
    tangent = np.zeros((2, 2, 2, 2) + determinant.shape)
    identity = np.eye(2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for ell in range(2):
                    tangent[i, j, k, ell] = (
                        MU * identity[i, k] * identity[j, ell]
                        + (MU - LAME_LAMBDA * np.log(determinant))
                        * inverse_transpose[i, ell]
                        * inverse_transpose[k, j]
                        + LAME_LAMBDA
                        * inverse_transpose[i, j]
                        * inverse_transpose[k, ell]
                    )
    return determinant, stress, tangent


def follower_direction(F):
    end_tangent = F[:, 1]
    norm = np.sqrt(dot(end_tangent, end_tangent))
    unit_tangent = end_tangent / norm[None, ...]
    return -unit_tangent, unit_tangent, norm


@LinearForm
def internal_force(v, w):
    return ddot(w.stress, grad(v))


@BilinearForm
def internal_tangent(du, v, w):
    return np.einsum("ijkl...,kl...,ij...->...", w.tangent, grad(du), grad(v))


@LinearForm
def boundary_load(v, w):
    return dot(w.direction, v) / HEIGHT


@BilinearForm
def follower_load_tangent(du, v, w):
    perturbation = grad(du)[:, 1]
    projection = perturbation - w.unit * dot(w.unit, perturbation)[None, ...]
    direction_derivative = -projection / w.norm[None, ...]
    return dot(direction_derivative, v) / HEIGHT


def main() -> None:
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, LENGTH, 49),
        np.linspace(-HEIGHT / 2.0, HEIGHT / 2.0, 9),
    )
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=4)
    right_facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], LENGTH))
    boundary_basis = FacetBasis(
        mesh, ElementVector(ElementTriP1()), facets=right_facets, intorder=4
    )
    fixed = basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    free = basis.complement_dofs(fixed)
    right = basis.get_dofs(lambda x: np.isclose(x[0], LENGTH))
    right_x, right_y = right.nodal["u^1"], right.nodal["u^2"]
    identity = np.eye(2)[:, :, None, None]
    dead_direction = np.zeros((2,) + boundary_basis.global_coordinates().shape[1:])
    dead_direction[1] = -1.0
    dead_vector = asm(boundary_load, boundary_basis, direction=dead_direction)

    def domain_state(displacement):
        F = identity + grad(basis.interpolate(displacement))
        return constitutive(F)

    def boundary_state(displacement):
        F = identity + grad(boundary_basis.interpolate(displacement))
        return F, follower_direction(F)

    def internal(displacement):
        _, stress, _ = domain_state(displacement)
        return asm(internal_force, basis, stress=stress)

    def stiffness(displacement):
        _, _, tangent = domain_state(displacement)
        return asm(internal_tangent, basis, tangent=tangent)

    def external(displacement, follower):
        if not follower:
            return dead_vector
        _, (direction, _, _) = boundary_state(displacement)
        return asm(boundary_load, boundary_basis, direction=direction)

    def external_tangent(displacement, follower):
        if not follower:
            from scipy.sparse import csr_matrix

            return csr_matrix((basis.N, basis.N))
        _, (_, unit, norm) = boundary_state(displacement)
        return asm(
            follower_load_tangent,
            boundary_basis,
            unit=unit,
            norm=norm,
        )

    load_path = np.linspace(0.0, 2.0e-3, 31)

    def solve_path(follower):
        displacement = np.zeros(basis.N)
        tip_x, tip_y = [0.0], [0.0]
        iterations = [0]
        saved = [(0.0, displacement.copy())]
        for step, load in enumerate(load_path[1:], 1):
            initial_norm = None
            for iteration in range(1, 26):
                residual = internal(displacement) - load * external(
                    displacement, follower
                )
                norm_residual = np.linalg.norm(residual[free])
                initial_norm = (
                    norm_residual if initial_norm is None else initial_norm
                )
                if norm_residual < max(1.0e-8, 1.0e-9 * initial_norm):
                    break
                tangent = stiffness(displacement) - load * external_tangent(
                    displacement, follower
                )
                increment = np.zeros_like(displacement)
                increment[free] = spsolve(
                    tangent[free][:, free], -residual[free]
                )
                alpha = 1.0
                while alpha > 1.0e-8:
                    trial = displacement + alpha * increment
                    if (
                        domain_state(trial)[0].min() > 0.0
                        and np.linalg.norm(
                            (
                                internal(trial)
                                - load * external(trial, follower)
                            )[free]
                        )
                        < norm_residual
                    ):
                        displacement = trial
                        break
                    alpha *= 0.5
                else:
                    raise RuntimeError("load-step line search failed")
            else:
                raise RuntimeError(f"load-step Newton failed at load={load}")
            tip_x.append(np.mean(displacement[right_x]))
            tip_y.append(np.mean(displacement[right_y]))
            iterations.append(iteration)
            if step in (10, 20, 30):
                saved.append((load, displacement.copy()))
        return displacement, np.asarray(tip_x), np.asarray(tip_y), iterations, saved

    dead, dead_x, dead_y, dead_iterations, dead_saved = solve_path(False)
    follower, follower_x, follower_y, follower_iterations, follower_saved = solve_path(
        True
    )

    final_load = load_path[-1]

    def follower_residual(displacement):
        return internal(displacement) - final_load * external(displacement, True)

    def follower_jacobian(displacement):
        return stiffness(displacement) - final_load * external_tangent(
            displacement, True
        )

    tangent_steps, tangent_errors = directional_derivative_errors(
        follower_residual,
        follower_jacobian,
        follower,
        free,
    )
    difference = np.linalg.norm(follower - dead) / np.linalg.norm(follower)
    print(
        f"final dead tip=({dead_x[-1]:+.5f}, {dead_y[-1]:+.5f}), "
        f"follower tip=({follower_x[-1]:+.5f}, {follower_y[-1]:+.5f}), "
        f"relative difference={difference:.3e}, "
        f"minimum tangent error={tangent_errors.min():.3e}"
    )
    assert difference > 1.0e-3
    assert tangent_errors.min() < 1.0e-7

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for label, saved, linestyle in (
        ("dead", dead_saved, "--"),
        ("follower", follower_saved, "-"),
    ):
        for index, (load, displacement) in enumerate(saved):
            nodal = displacement[basis.nodal_dofs]
            deformed = mesh.p + nodal
            triangulation = mtri.Triangulation(
                deformed[0], deformed[1], mesh.t.T
            )
            axes[0, 0].triplot(
                triangulation,
                linestyle=linestyle,
                linewidth=0.45,
                label=f"{label}, P={load:.4f}" if index == len(saved) - 1 else None,
            )
    axes[0, 0].set(title="Finite-deformation cantilever", xlabel="x", ylabel="y", aspect="equal")
    axes[0, 0].legend()

    axes[0, 1].plot(-dead_y, load_path, "o-", label="dead load")
    axes[0, 1].plot(-follower_y, load_path, "s-", label="follower load")
    axes[0, 1].set(title="Load–deflection response", xlabel="downward tip displacement", ylabel="load")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(dead_x, dead_y, "o-", label="dead load")
    axes[1, 0].plot(follower_x, follower_y, "s-", label="follower load")
    axes[1, 0].set(title="Tip trajectory", xlabel="horizontal displacement", ylabel="vertical displacement")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].loglog(tangent_steps, tangent_errors, "o-", label="total follower tangent")
    axes[1, 1].loglog(
        tangent_steps,
        tangent_errors[0] * (tangent_steps / tangent_steps[0]) ** 2,
        "k--",
        label=r"$O(h^2)$",
    )
    axes[1, 1].invert_xaxis()
    axes[1, 1].set(title="Internal + external tangent check", xlabel="difference step", ylabel="relative error")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend()
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
