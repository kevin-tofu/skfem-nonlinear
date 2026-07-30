"""Two-way steady thermo-mechanical coupling: staggered versus monolithic."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, LinearForm, MeshTri, asm
from skfem.helpers import ddot, div, dot, grad, sym_grad, trace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.nonlinear_verification import directional_derivative_errors


LAME_LAMBDA, LAME_MU = 2.0, 1.0
THERMAL_EXPANSION = 0.15
THERMAL_SOFTENING = 0.8
TEMPERATURE_CONDUCTIVITY = 1.5
STRAIN_CONDUCTIVITY = 2.0
I2 = np.eye(2)[:, :, None, None]


def material(temperature, displacement):
    strain = sym_grad(displacement)
    scale = 1.0 / (1.0 + THERMAL_SOFTENING * temperature)
    elastic_strain = strain - THERMAL_EXPANSION * temperature[None, None, ...] * I2
    trace_elastic = trace(elastic_strain)
    base_stress = (
        2.0 * LAME_MU * elastic_strain
        + LAME_LAMBDA * trace_elastic[None, None, ...] * I2
    )
    stress = scale[None, None, ...] * base_stress
    scale_derivative = -THERMAL_SOFTENING * scale**2
    thermal_stress_derivative = -2.0 * THERMAL_EXPANSION * (
        LAME_MU + LAME_LAMBDA
    ) * I2
    stress_temperature_derivative = (
        scale_derivative[None, None, ...] * base_stress
        + scale[None, None, ...] * thermal_stress_derivative
    )
    conductivity = (1.0 + TEMPERATURE_CONDUCTIVITY * temperature**2) * np.exp(
        STRAIN_CONDUCTIVITY * div(displacement)
    )
    conductivity_temperature_derivative = conductivity * (
        2.0 * TEMPERATURE_CONDUCTIVITY * temperature
        / (1.0 + TEMPERATURE_CONDUCTIVITY * temperature**2)
    )
    return (
        stress,
        scale,
        stress_temperature_derivative,
        conductivity,
        conductivity_temperature_derivative,
    )


@LinearForm
def mechanical_residual(v, w):
    return ddot(w.stress, sym_grad(v))


@LinearForm
def thermal_residual(q, w):
    return w.conductivity * dot(grad(w.temperature), grad(q))


@BilinearForm
def mechanical_displacement_tangent(du, v, w):
    eu, ev = sym_grad(du), sym_grad(v)
    return w.scale * (
        2.0 * LAME_MU * ddot(eu, ev)
        + LAME_LAMBDA * trace(eu) * trace(ev)
    )


@BilinearForm
def mechanical_temperature_tangent(dtemperature, v, w):
    return dtemperature * ddot(w.stress_temperature_derivative, sym_grad(v))


@BilinearForm
def thermal_displacement_tangent(du, q, w):
    return (
        STRAIN_CONDUCTIVITY
        * w.conductivity
        * div(du)
        * dot(grad(w.temperature), grad(q))
    )


@BilinearForm
def thermal_temperature_tangent(dtemperature, q, w):
    return (
        w.conductivity * dot(grad(dtemperature), grad(q))
        + w.conductivity_temperature_derivative
        * dtemperature
        * dot(grad(w.temperature), grad(q))
    )


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 25), np.linspace(0.0, 1.0, 25))
    displacement_basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=5)
    temperature_basis = Basis(mesh, ElementTriP1(), intorder=5)
    nu, nt = displacement_basis.N, temperature_basis.N

    fixed_u = displacement_basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all()
    fixed_t = np.unique(
        np.concatenate(
            (
                temperature_basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all(),
                temperature_basis.get_dofs(lambda x: np.isclose(x[0], 1.0)).all(),
            )
        )
    )
    right_t = temperature_basis.get_dofs(lambda x: np.isclose(x[0], 1.0)).all()
    free_u = displacement_basis.complement_dofs(fixed_u)
    free_t = temperature_basis.complement_dofs(fixed_t)
    free = np.concatenate((free_u, nu + free_t))

    initial_u = np.zeros(nu)
    initial_t = temperature_basis.doflocs[0].copy()
    initial_t[fixed_t] = 0.0
    initial_t[right_t] = 1.0

    def fields(state):
        displacement = displacement_basis.interpolate(state[:nu])
        temperature = temperature_basis.interpolate(state[nu:])
        return displacement, temperature, material(temperature, displacement)

    def residual(state):
        displacement, temperature, values = fields(state)
        stress, _, _, conductivity, _ = values
        mechanical = asm(
            mechanical_residual, displacement_basis, stress=stress
        )
        thermal = asm(
            thermal_residual,
            temperature_basis,
            conductivity=conductivity,
            temperature=temperature,
        )
        return np.concatenate((mechanical, thermal))

    def blocks(state):
        displacement, temperature, values = fields(state)
        (
            _,
            scale,
            stress_temperature_derivative,
            conductivity,
            conductivity_temperature_derivative,
        ) = values
        common = dict(
            scale=scale,
            stress_temperature_derivative=stress_temperature_derivative,
            conductivity=conductivity,
            conductivity_temperature_derivative=conductivity_temperature_derivative,
            temperature=temperature,
        )
        Kuu = asm(
            mechanical_displacement_tangent, displacement_basis, **common
        )
        KuT = asm(
            mechanical_temperature_tangent,
            temperature_basis,
            displacement_basis,
            **common,
        )
        KTu = asm(
            thermal_displacement_tangent,
            displacement_basis,
            temperature_basis,
            **common,
        )
        KTT = asm(
            thermal_temperature_tangent, temperature_basis, **common
        )
        return Kuu, KuT, KTu, KTT

    def full_jacobian(state):
        Kuu, KuT, KTu, KTT = blocks(state)
        return bmat([[Kuu, KuT], [KTu, KTT]], format="csr")

    initial_state = np.concatenate((initial_u, initial_t))

    def solve_monolithic():
        state = initial_state.copy()
        history = []
        initial_norm = None
        for iteration in range(1, 31):
            r = residual(state)
            norm_r = np.linalg.norm(r[free])
            history.append(norm_r)
            initial_norm = norm_r if initial_norm is None else initial_norm
            if norm_r < max(1.0e-11, 1.0e-10 * initial_norm):
                break
            J = full_jacobian(state)
            increment = np.zeros_like(state)
            increment[free] = spsolve(J[free][:, free], -r[free])
            alpha = 1.0
            while alpha > 1.0e-8:
                trial = state + alpha * increment
                if (
                    np.min(1.0 + THERMAL_SOFTENING * trial[nu:]) > 0.0
                    and np.linalg.norm(residual(trial)[free]) < norm_r
                ):
                    state = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("monolithic line search failed")
        else:
            raise RuntimeError("monolithic Newton failed")
        return state, np.asarray(history)

    def solve_staggered():
        state = initial_state.copy()
        history = []
        for iteration in range(1, 101):
            # Nonlinear thermal subproblem with displacement held fixed.
            for _ in range(15):
                r = residual(state)
                if np.linalg.norm(r[nu + free_t]) < 1.0e-11:
                    break
                KTT = blocks(state)[3]
                increment_t = spsolve(
                    KTT[free_t][:, free_t], -r[nu + free_t]
                )
                state[nu + free_t] += increment_t

            # Mechanical subproblem is linear for a fixed temperature.
            r = residual(state)
            Kuu = blocks(state)[0]
            state[free_u] += spsolve(Kuu[free_u][:, free_u], -r[free_u])
            norm_r = np.linalg.norm(residual(state)[free])
            history.append(norm_r)
            if norm_r < 1.0e-9:
                break
        else:
            raise RuntimeError("staggered iteration did not converge")
        return state, np.asarray(history)

    monolithic, monolithic_history = solve_monolithic()
    staggered, staggered_history = solve_staggered()
    relative_difference = np.linalg.norm(monolithic - staggered) / np.linalg.norm(monolithic)
    tangent_steps, tangent_errors = directional_derivative_errors(
        residual, full_jacobian, monolithic, free
    )
    print(
        f"monolithic iterations={len(monolithic_history)}, "
        f"staggered iterations={len(staggered_history)}, "
        f"relative difference={relative_difference:.3e}, "
        f"minimum tangent error={tangent_errors.min():.3e}"
    )
    assert relative_difference < 1.0e-7
    assert tangent_errors.min() < 1.0e-7

    displacement = monolithic[:nu][displacement_basis.nodal_dofs]
    temperature = monolithic[nu:]
    deformed = mesh.p + displacement
    triangulation = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    deformed_triangulation = mtri.Triangulation(deformed[0], deformed[1], mesh.t.T)
    displacement_magnitude = np.sqrt(np.sum(displacement**2, axis=0))
    stress = fields(monolithic)[2][0]
    element_stress = stress.mean(axis=3)
    von_mises = np.sqrt(
        element_stress[0, 0] ** 2
        - element_stress[0, 0] * element_stress[1, 1]
        + element_stress[1, 1] ** 2
        + 3.0 * element_stress[0, 1] ** 2
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    temperature_image = axes[0, 0].tricontourf(
        triangulation, temperature, levels=24, cmap="inferno"
    )
    fig.colorbar(temperature_image, ax=axes[0, 0], label="temperature")
    axes[0, 0].set(title="Coupled temperature", xlabel="x", ylabel="y", aspect="equal")

    displacement_image = axes[0, 1].tripcolor(
        deformed_triangulation, displacement_magnitude, shading="gouraud", cmap="viridis"
    )
    fig.colorbar(displacement_image, ax=axes[0, 1], label="displacement magnitude")
    axes[0, 1].triplot(deformed_triangulation, color="white", linewidth=0.25, alpha=0.5)
    axes[0, 1].set(title="Thermally deformed body", xlabel="x", ylabel="y", aspect="equal")

    stress_image = axes[0, 2].tripcolor(
        deformed_triangulation, facecolors=von_mises, shading="flat", cmap="magma"
    )
    fig.colorbar(stress_image, ax=axes[0, 2], label="von Mises stress")
    axes[0, 2].set(title="Thermal stress", xlabel="x", ylabel="y", aspect="equal")

    axes[1, 0].semilogy(
        range(1, len(monolithic_history) + 1), monolithic_history, "o-", label="monolithic"
    )
    axes[1, 0].semilogy(
        range(1, len(staggered_history) + 1), staggered_history, "s-", label="staggered"
    )
    axes[1, 0].set(title="Coupling convergence", xlabel="iteration", ylabel="coupled residual")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend()

    difference_image = axes[1, 1].tricontourf(
        triangulation,
        monolithic[nu:] - staggered[nu:],
        levels=21,
        cmap="PiYG",
    )
    fig.colorbar(difference_image, ax=axes[1, 1], label="monolithic − staggered")
    axes[1, 1].set(title="Temperature-method difference", xlabel="x", ylabel="y", aspect="equal")

    axes[1, 2].loglog(tangent_steps, tangent_errors, "o-", label="block tangent")
    axes[1, 2].loglog(
        tangent_steps,
        tangent_errors[0] * (tangent_steps / tangent_steps[0]) ** 2,
        "k--",
        label=r"$O(h^2)$",
    )
    axes[1, 2].invert_xaxis()
    axes[1, 2].set(title="Tangent verification", xlabel="difference step", ylabel="relative error")
    axes[1, 2].grid(True, which="both", alpha=0.3)
    axes[1, 2].legend()
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
