"""Steady lid-driven cavity: Picard versus Newton linearization."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, LinearForm
from skfem import MeshTri, asm
from skfem.helpers import ddot, div, dot, grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.nonlinear_verification import directional_derivative_errors


REYNOLDS = 100.0
VISCOSITY = 1.0 / REYNOLDS


@LinearForm
def momentum_residual(v, w):
    convection = np.einsum("j...,ij...,i...->...", w.velocity, grad(w.velocity), v)
    return VISCOSITY * ddot(grad(w.velocity), grad(v)) + convection - w.pressure * div(v)


@LinearForm
def continuity_residual(q, w):
    return q * div(w.velocity)


@BilinearForm
def newton_velocity_block(du, v, w):
    convection = np.einsum("j...,ij...,i...->...", w.velocity, grad(du), v)
    convection_increment = np.einsum("j...,ij...,i...->...", du, grad(w.velocity), v)
    return VISCOSITY * ddot(grad(du), grad(v)) + convection + convection_increment


@BilinearForm
def picard_velocity_block(du, v, w):
    convection = np.einsum("j...,ij...,i...->...", w.velocity, grad(du), v)
    return VISCOSITY * ddot(grad(du), grad(v)) + convection


@BilinearForm
def pressure_to_momentum(dp, v, w):
    return -dp * div(v)


@BilinearForm
def velocity_to_continuity(du, q, w):
    return q * div(du)


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 17), np.linspace(0.0, 1.0, 17))
    velocity_basis = Basis(mesh, ElementVector(ElementTriP2()), intorder=5)
    pressure_basis = Basis(mesh, ElementTriP1(), intorder=5)
    nv, np_ = velocity_basis.N, pressure_basis.N

    walls = velocity_basis.get_dofs().all()
    free_velocity = velocity_basis.complement_dofs(walls)
    free_pressure = np.arange(1, np_)  # p[0] = 0 removes the pressure nullspace.
    free = np.concatenate((free_velocity, nv + free_pressure))
    fixed = np.setdiff1d(np.arange(nv + np_), free)

    prescribed = np.zeros(nv + np_)
    top_x = velocity_basis.get_dofs(lambda x: np.isclose(x[1], 1.0)).all("u^1")
    top_x = top_x[
        (velocity_basis.doflocs[0, top_x] > 0.0)
        & (velocity_basis.doflocs[0, top_x] < 1.0)
    ]
    prescribed[top_x] = 1.0
    zero_pp = csr_matrix((np_, np_))

    def fields(state):
        return (
            velocity_basis.interpolate(state[:nv]),
            pressure_basis.interpolate(state[nv:]),
        )

    def residual(state):
        velocity, pressure = fields(state)
        momentum = asm(
            momentum_residual,
            velocity_basis,
            velocity=velocity,
            pressure=pressure,
        )
        continuity = asm(continuity_residual, pressure_basis, velocity=velocity)
        return np.concatenate((momentum, continuity))

    def block_matrix(state, newton):
        velocity, _ = fields(state)
        velocity_form = newton_velocity_block if newton else picard_velocity_block
        Kvv = asm(velocity_form, velocity_basis, velocity=velocity)
        Kvp = asm(pressure_to_momentum, pressure_basis, velocity_basis)
        Kpv = asm(velocity_to_continuity, velocity_basis, pressure_basis)
        return bmat([[Kvv, Kvp], [Kpv, zero_pp]], format="csr")

    # Stokes solution: Picard matrix with zero advecting velocity.
    stokes_state = prescribed.copy()
    stokes_matrix = block_matrix(np.zeros_like(stokes_state), newton=False)
    stokes_state[free] = spsolve(
        stokes_matrix[free][:, free],
        -(stokes_matrix[free][:, fixed] @ prescribed[fixed]),
    )

    def solve_picard():
        state = stokes_state.copy()
        history = []
        relaxation = 0.7
        for iteration in range(1, 151):
            norm_r = np.linalg.norm(residual(state)[free])
            history.append(norm_r)
            if norm_r < 1.0e-9:
                break
            matrix = block_matrix(state, newton=False)
            candidate = prescribed.copy()
            candidate[free] = spsolve(
                matrix[free][:, free],
                -(matrix[free][:, fixed] @ prescribed[fixed]),
            )
            state = (1.0 - relaxation) * state + relaxation * candidate
            state[fixed] = prescribed[fixed]
        else:
            raise RuntimeError("Picard iteration did not converge")
        return state, np.asarray(history)

    def solve_newton():
        state = stokes_state.copy()
        history = []
        for iteration in range(1, 31):
            r = residual(state)
            norm_r = np.linalg.norm(r[free])
            history.append(norm_r)
            if norm_r < 1.0e-9:
                break
            matrix = block_matrix(state, newton=True)
            increment = np.zeros_like(state)
            increment[free] = spsolve(matrix[free][:, free], -r[free])
            alpha = 1.0
            while alpha > 1.0e-8:
                trial = state + alpha * increment
                if np.linalg.norm(residual(trial)[free]) < norm_r:
                    state = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("Newton line search failed")
        else:
            raise RuntimeError("Newton iteration did not converge")
        return state, np.asarray(history)

    picard_state, picard_history = solve_picard()
    newton_state, newton_history = solve_newton()
    relative_difference = np.linalg.norm(
        picard_state[:nv] - newton_state[:nv]
    ) / np.linalg.norm(newton_state[:nv])
    steps, tangent_errors = directional_derivative_errors(
        residual,
        lambda state: block_matrix(state, newton=True),
        newton_state,
        free,
    )
    print(
        f"Re={REYNOLDS:g}, Picard iterations={len(picard_history)}, "
        f"Newton iterations={len(newton_history)}, "
        f"relative velocity difference={relative_difference:.3e}, "
        f"minimum tangent error={tangent_errors.min():.3e}"
    )
    assert relative_difference < 1.0e-7
    assert tangent_errors.min() < 1.0e-7

    nodal_velocity = newton_state[:nv][velocity_basis.nodal_dofs]
    speed = np.sqrt(np.sum(nodal_velocity**2, axis=0))
    pressure = newton_state[nv:]
    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    speed_image = axes[0, 0].tricontourf(triangulation, speed, levels=24, cmap="viridis")
    fig.colorbar(speed_image, ax=axes[0, 0], label="speed")
    skip = 2
    axes[0, 0].quiver(
        mesh.p[0, ::skip],
        mesh.p[1, ::skip],
        nodal_velocity[0, ::skip],
        nodal_velocity[1, ::skip],
        color="white",
        scale=8,
    )
    axes[0, 0].set(title=f"Lid-driven cavity, Re={REYNOLDS:g}", xlabel="x", ylabel="y", aspect="equal")

    pressure_image = axes[0, 1].tricontourf(
        triangulation, pressure, levels=24, cmap="coolwarm"
    )
    fig.colorbar(pressure_image, ax=axes[0, 1], label="pressure")
    axes[0, 1].set(title="Pressure field", xlabel="x", ylabel="y", aspect="equal")

    axes[0, 2].loglog(steps, tangent_errors, "o-", label="Newton tangent")
    axes[0, 2].loglog(
        steps,
        tangent_errors[0] * (steps / steps[0]) ** 2,
        "k--",
        label=r"$O(h^2)$",
    )
    axes[0, 2].invert_xaxis()
    axes[0, 2].set(title="Tangent verification", xlabel="difference step", ylabel="relative error")
    axes[0, 2].grid(True, which="both", alpha=0.3)
    axes[0, 2].legend()

    axes[1, 0].semilogy(range(1, len(picard_history) + 1), picard_history, "o-", label="Picard")
    axes[1, 0].semilogy(range(1, len(newton_history) + 1), newton_history, "s-", label="Newton")
    axes[1, 0].set(title="Nonlinear convergence", xlabel="iteration", ylabel="residual norm")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend()

    for component, coordinate, label in ((0, 0, r"$u_x$ at $x=0.5$"), (1, 1, r"$u_y$ at $y=0.5$")):
        mask = np.isclose(mesh.p[coordinate], 0.5)
        varying_coordinate = 1 - coordinate
        order = np.argsort(mesh.p[varying_coordinate, mask])
        axes[1, 1].plot(
            nodal_velocity[component, mask][order],
            mesh.p[varying_coordinate, mask][order],
            "o-",
            label=label,
        )
    axes[1, 1].axvline(0.0, color="0.5", linewidth=0.8)
    axes[1, 1].set(title="Cavity centre-line velocities", xlabel="velocity", ylabel="coordinate")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.05,
        0.75,
        f"Reynolds number: {REYNOLDS:g}\n"
        f"Picard iterations: {len(picard_history)}\n"
        f"Newton iterations: {len(newton_history)}\n"
        f"Velocity difference: {relative_difference:.2e}\n"
        f"Best tangent error: {tangent_errors.min():.2e}",
        fontsize=13,
        va="top",
        family="monospace",
    )
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
