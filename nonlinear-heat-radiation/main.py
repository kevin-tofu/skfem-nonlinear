"""Temperature-dependent conduction with nonlinear radiation on one boundary."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.nonlinear_verification import directional_derivative_errors


SOURCE = 5.0
RADIATION = 2.0
AMBIENT_ABSOLUTE_TEMPERATURE = 1.0


def conductivity(temperature):
    return 1.0 + 0.5 * temperature**2


@LinearForm
def domain_residual(v, w):
    return conductivity(w.temperature) * dot(grad(w.temperature), grad(v)) - SOURCE * v


@BilinearForm
def domain_tangent(dtemperature, v, w):
    return (
        conductivity(w.temperature) * dot(grad(dtemperature), grad(v))
        + w.temperature * dtemperature * dot(grad(w.temperature), grad(v))
    )


@LinearForm
def radiation_residual(v, w):
    absolute_temperature = AMBIENT_ABSOLUTE_TEMPERATURE + w.temperature
    return RADIATION * (absolute_temperature**4 - AMBIENT_ABSOLUTE_TEMPERATURE**4) * v


@BilinearForm
def radiation_tangent(dtemperature, v, w):
    absolute_temperature = AMBIENT_ABSOLUTE_TEMPERATURE + w.temperature
    return 4.0 * RADIATION * absolute_temperature**3 * dtemperature * v


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 41), np.linspace(0.0, 1.0, 25))
    basis = Basis(mesh, ElementTriP1(), intorder=5)
    right_facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 1.0))
    radiation_basis = FacetBasis(mesh, ElementTriP1(), facets=right_facets, intorder=5)
    fixed = basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    free = basis.complement_dofs(fixed)
    temperature = np.zeros(basis.N)

    def residual(state):
        result = asm(
            domain_residual, basis, temperature=basis.interpolate(state)
        )
        result += asm(
            radiation_residual,
            radiation_basis,
            temperature=radiation_basis.interpolate(state),
        )
        return result

    def jacobian(state):
        result = asm(
            domain_tangent, basis, temperature=basis.interpolate(state)
        )
        result += asm(
            radiation_tangent,
            radiation_basis,
            temperature=radiation_basis.interpolate(state),
        )
        return result

    residual_history = []
    initial_norm = None
    for iteration in range(1, 26):
        r = residual(temperature)
        norm_r = np.linalg.norm(r[free])
        residual_history.append(norm_r)
        initial_norm = norm_r if initial_norm is None else initial_norm
        print(f"Newton {iteration:2d}: residual={norm_r:.3e}")
        if norm_r < max(1.0e-11, 1.0e-10 * initial_norm):
            break
        J = jacobian(temperature)
        increment = np.zeros_like(temperature)
        increment[free] = spsolve(J[free][:, free], -r[free])

        alpha = 1.0
        while alpha > 1.0e-8:
            trial = temperature + alpha * increment
            if (
                np.min(AMBIENT_ABSOLUTE_TEMPERATURE + trial) > 0.0
                and np.linalg.norm(residual(trial)[free]) < norm_r
            ):
                temperature = trial
                break
            alpha *= 0.5
        else:
            raise RuntimeError("Newton line search failed")
    else:
        raise RuntimeError("Newton iteration did not converge")

    steps, tangent_errors = directional_derivative_errors(
        residual, jacobian, temperature, free
    )
    minimum_tangent_error = tangent_errors.min()
    print(
        f"temperature=[{temperature.min():.6f}, {temperature.max():.6f}], "
        f"minimum tangent error={minimum_tangent_error:.3e}"
    )
    assert np.isfinite(temperature).all() and temperature.max() > 0.0
    assert minimum_tangent_error < 1.0e-7

    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    right_dofs = basis.get_dofs(lambda x: np.isclose(x[0], 1.0)).all()
    order = np.argsort(basis.doflocs[1, right_dofs])
    boundary_temperature = temperature[right_dofs][order]
    radiation_flux = RADIATION * (
        (AMBIENT_ABSOLUTE_TEMPERATURE + boundary_temperature) ** 4
        - AMBIENT_ABSOLUTE_TEMPERATURE**4
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    image = axes[0, 0].tricontourf(triangulation, temperature, levels=24, cmap="inferno")
    fig.colorbar(image, ax=axes[0, 0], label="temperature rise")
    axes[0, 0].set(title="Nonlinear temperature field", xlabel="x", ylabel="y", aspect="equal")
    axes[0, 1].plot(radiation_flux, basis.doflocs[1, right_dofs][order], "o-")
    axes[0, 1].set(title="Radiation on right boundary", xlabel="outward heat flux", ylabel="y")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].semilogy(range(1, len(residual_history) + 1), residual_history, "o-")
    axes[1, 0].set(title="Newton convergence", xlabel="iteration", ylabel="residual norm")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 1].loglog(steps, tangent_errors, "o-", label="centred difference error")
    axes[1, 1].loglog(steps, tangent_errors[0] * (steps / steps[0]) ** 2, "--", label=r"$O(h^2)$")
    axes[1, 1].invert_xaxis()
    axes[1, 1].set(title="Consistent-tangent verification", xlabel="difference step", ylabel="relative error")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend()
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
