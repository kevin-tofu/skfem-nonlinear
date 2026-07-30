"""Nearly incompressible neo-Hookean elasticity in mixed displacement-pressure form.

A P2 displacement / P1 pressure Taylor--Hood pair avoids volumetric locking.
The unit square is stretched horizontally using load stepping and damped
Newton iterations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, LinearForm
from skfem import MeshTri, asm
from skfem.helpers import ddot, grad


SHEAR = 1.0
BULK = 1.0e4


def kinematics(displacement):
    F = np.eye(2)[:, :, None, None] + grad(displacement)
    moved = np.moveaxis(F, (0, 1), (-2, -1))
    determinant = np.linalg.det(moved)
    inverse_transpose = np.swapaxes(np.linalg.inv(moved), -1, -2)
    inverse_transpose = np.moveaxis(inverse_transpose, (-2, -1), (0, 1))
    cofactor = determinant[None, None, ...] * inverse_transpose
    return F, determinant, inverse_transpose, cofactor


@LinearForm
def displacement_residual(v, w):
    stress = SHEAR * (w.F - w.FinvT) + w.pressure * w.cofactor
    return ddot(stress, grad(v))


@LinearForm
def pressure_residual(q, w):
    return q * (w.determinant - 1.0 - w.pressure / BULK)


@BilinearForm
def tangent_uu(du, v, w):
    gu, gv = grad(du), grad(v)
    geometric = np.einsum(
        "iL...,kJ...,kL...,iJ...->...", w.FinvT, w.FinvT, gu, gv
    )
    cofactor_part = np.einsum(
        "iJ...,kL...,kL...,iJ...->...", w.FinvT, w.FinvT, gu, gv
    ) - np.einsum(
        "iL...,kJ...,kL...,iJ...->...", w.FinvT, w.FinvT, gu, gv
    )
    return SHEAR * (ddot(gu, gv) + geometric) + w.pressure * w.determinant * cofactor_part


@BilinearForm
def tangent_up(dp, v, w):
    return dp * ddot(w.cofactor, grad(v))


@BilinearForm
def tangent_pu(du, q, w):
    return q * ddot(w.cofactor, grad(du))


@BilinearForm
def tangent_pp(dp, q, w):
    return -dp * q / BULK


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 13), np.linspace(0.0, 1.0, 13))
    displacement_basis = Basis(mesh, ElementVector(ElementTriP2()), intorder=5)
    pressure_basis = Basis(mesh, ElementTriP1(), intorder=5)
    nu, np_ = displacement_basis.N, pressure_basis.N

    left_x = displacement_basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all("u^1")
    right_x = displacement_basis.get_dofs(lambda x: np.isclose(x[0], 1.0)).all("u^1")
    left_y = displacement_basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all("u^2")
    fixed_u = np.unique(np.concatenate((left_x, left_y, right_x)))
    free_u = displacement_basis.complement_dofs(fixed_u)
    free = np.concatenate((free_u, nu + np.arange(np_)))

    u = np.zeros(nu)
    pressure = np.zeros(np_)
    displacement_history, reaction_history = [], []

    def fields(u_vector, p_vector):
        F, determinant, inverse_transpose, cofactor = kinematics(
            displacement_basis.interpolate(u_vector)
        )
        pressure_qp = pressure_basis.interpolate(p_vector)
        return F, determinant, inverse_transpose, cofactor, pressure_qp

    def residual(u_vector, p_vector):
        F, determinant, inverse_transpose, cofactor, pressure_qp = fields(u_vector, p_vector)
        ru = asm(
            displacement_residual,
            displacement_basis,
            F=F,
            FinvT=inverse_transpose,
            cofactor=cofactor,
            pressure=pressure_qp,
        )
        rp = asm(
            pressure_residual,
            pressure_basis,
            determinant=determinant,
            pressure=pressure_qp,
        )
        return np.concatenate((ru, rp))

    for load_step, imposed in enumerate(np.linspace(0.02, 0.20, 10), 1):
        u[right_x] = imposed
        initial_residual = None
        for iteration in range(1, 26):
            F, determinant, inverse_transpose, cofactor, pressure_qp = fields(u, pressure)
            if determinant.min() <= 0.0:
                raise RuntimeError("element inversion in accepted state")
            r = residual(u, pressure)
            norm_r = np.linalg.norm(r[free])
            initial_residual = norm_r if initial_residual is None else initial_residual
            if norm_r < max(1.0e-10, 1.0e-9 * initial_residual):
                break

            common = dict(
                FinvT=inverse_transpose,
                cofactor=cofactor,
                determinant=determinant,
                pressure=pressure_qp,
            )
            Kuu = asm(tangent_uu, displacement_basis, **common)
            Kup = asm(tangent_up, pressure_basis, displacement_basis, **common)
            Kpu = asm(tangent_pu, displacement_basis, pressure_basis, **common)
            Kpp = asm(tangent_pp, pressure_basis)
            K = bmat([[Kuu, Kup], [Kpu, Kpp]], format="csr")
            increment = np.zeros(nu + np_)
            increment[free] = spsolve(K[free][:, free], -r[free])

            alpha = 1.0
            while alpha > 1.0e-8:
                trial_u = u + alpha * increment[:nu]
                trial_p = pressure + alpha * increment[nu:]
                trial_determinant = fields(trial_u, trial_p)[1]
                if (
                    trial_determinant.min() > 0.0
                    and np.linalg.norm(residual(trial_u, trial_p)[free]) < norm_r
                ):
                    u, pressure = trial_u, trial_p
                    break
                alpha *= 0.5
            else:
                raise RuntimeError(f"line search failed at step {load_step}")
        else:
            raise RuntimeError(f"Newton failed at step {load_step}")

        final_residual = residual(u, pressure)
        reaction = final_residual[right_x].sum()
        determinant = fields(u, pressure)[1]
        displacement_history.append(imposed)
        reaction_history.append(reaction)
        print(
            f"step={load_step:2d}, ux={imposed:.3f}, Newton={iteration:2d}, "
            f"reaction={reaction:.6f}, J=[{determinant.min():.6f}, "
            f"{determinant.max():.6f}], mean(J)={determinant.mean():.6f}, "
            f"p=[{pressure.min():.4f}, {pressure.max():.4f}]"
        )

    assert np.isfinite(u).all() and np.isfinite(pressure).all()
    assert fields(u, pressure)[1].min() > 0.0
    # Incompressibility is imposed weakly, so pointwise J may deviate near the
    # clamped corners while its domain average remains close to one.
    assert abs(fields(u, pressure)[1].mean() - 1.0) < 2.0e-3

    # P2 contains vertex and edge DOFs; nodal_dofs selects the vertex values.
    nodal_u = u[displacement_basis.nodal_dofs]
    deformed = mesh.p + nodal_u
    deformed_tri = mtri.Triangulation(deformed[0], deformed[1], mesh.t.T)
    element_J = fields(u, pressure)[1].mean(axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].triplot(deformed_tri, color="tab:blue", linewidth=0.55)
    axes[0, 0].set(title="Deformed Taylor–Hood mesh", xlabel="x", ylabel="y", aspect="equal")
    pressure_image = axes[0, 1].tricontourf(deformed_tri, pressure, levels=20, cmap="coolwarm")
    fig.colorbar(pressure_image, ax=axes[0, 1], label="pressure")
    axes[0, 1].set(title="Pressure field", xlabel="x", ylabel="y", aspect="equal")
    volume_image = axes[1, 0].tripcolor(
        deformed_tri, facecolors=element_J, shading="flat", cmap="viridis"
    )
    fig.colorbar(volume_image, ax=axes[1, 0], label=r"$\det F$")
    axes[1, 0].set(title="Local volume ratio", xlabel="x", ylabel="y", aspect="equal")
    axes[1, 1].plot(displacement_history, reaction_history, "o-", label="reaction")
    axes[1, 1].set(
        title=f"Nearly incompressible response (κ/μ={BULK / SHEAR:.0e})",
        xlabel="prescribed $u_x$",
        ylabel="reaction",
    )
    axes[1, 1].grid(True, alpha=0.3)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
