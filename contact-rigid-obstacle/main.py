"""Frictionless contact of an elastic square with a curved rigid obstacle.

Two nodal contact algorithms are compared:

* penalty regularization;
* primal-dual active set (exact nodal non-penetration).

The obstacle is below the bottom edge and is closest at the centre, so contact
starts near x=0.5 and spreads as the top displacement increases.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from scipy.sparse import diags
from skfem import Basis, ElementTriP1, ElementVector, MeshTri, asm, condense, solve
from skfem.models.elasticity import linear_elasticity


YOUNG, POISSON = 1000.0, 0.30
LAME_LAMBDA = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
LAME_MU = YOUNG / (2.0 * (1.0 + POISSON))


def obstacle_gap(x):
    """Initial vertical distance between the flat body and curved obstacle."""
    return 0.015 + 0.12 * (x - 0.5) ** 2


def constrained_solve(K, rhs, prescribed, dofs):
    return solve(*condense(K, rhs, x=prescribed, D=np.unique(dofs)))


def solve_penalty(K, basis, top_y, anchor_x, bottom_y, gaps, imposed, previous):
    penalty = 2.0e5
    u = previous.copy()
    prescribed = np.zeros(basis.N)
    prescribed[top_y] = imposed
    base_dofs = np.concatenate((top_y, anchor_x))

    active_old = None
    for iteration in range(1, 31):
        active = u[bottom_y] + gaps < 0.0
        diagonal = np.zeros(basis.N)
        rhs = np.zeros(basis.N)
        contact_dofs = bottom_y[active]
        diagonal[contact_dofs] = penalty
        rhs[contact_dofs] = -penalty * gaps[active]
        u = constrained_solve(K + diags(diagonal), rhs, prescribed, base_dofs)
        if active_old is not None and np.array_equal(active, active_old):
            break
        active_old = active
    else:
        raise RuntimeError("penalty active-set iteration did not converge")

    penetration = np.minimum(u[bottom_y] + gaps, 0.0)
    contact_force = -penalty * penetration
    return u, contact_force, iteration


def solve_active_set(K, basis, top_y, anchor_x, bottom_y, gaps, imposed, previous):
    """Primal-dual active set for g >= 0, lambda >= 0, g*lambda = 0."""
    scale = 2.0e5
    u = previous.copy()
    multiplier = np.zeros(bottom_y.size)
    prescribed = np.zeros(basis.N)
    prescribed[top_y] = imposed

    for iteration in range(1, 31):
        gap = u[bottom_y] + gaps
        active = multiplier - scale * gap > 0.0
        prescribed[bottom_y[active]] = -gaps[active]
        dofs = np.concatenate((top_y, anchor_x, bottom_y[active]))
        u_new = constrained_solve(K, np.zeros(basis.N), prescribed, dofs)
        reaction = K @ u_new
        multiplier_new = np.zeros_like(multiplier)
        multiplier_new[active] = reaction[bottom_y[active]]

        gap_new = u_new[bottom_y] + gaps
        active_new = multiplier_new - scale * gap_new > 0.0
        u, multiplier = u_new, multiplier_new
        if np.array_equal(active_new, active):
            break
    else:
        raise RuntimeError("primal-dual active-set iteration did not converge")

    return u, multiplier, iteration


def main() -> None:
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0, 33), np.linspace(0.0, 1.0, 17))
    basis = Basis(mesh, ElementVector(ElementTriP1()))
    K = asm(linear_elasticity(LAME_LAMBDA, LAME_MU), basis)

    top = basis.get_dofs(lambda x: np.isclose(x[1], 1.0))
    bottom = basis.get_dofs(lambda x: np.isclose(x[1], 0.0))
    top_y = top.nodal["u^2"]
    bottom_y = bottom.nodal["u^2"]
    anchor_x = basis.get_dofs(
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 1.0)
    ).nodal["u^1"]
    gaps = obstacle_gap(basis.doflocs[0, bottom_y])

    penalty_u = np.zeros(basis.N)
    active_u = np.zeros(basis.N)
    imposed_history, force_history = [], []
    for imposed in -np.linspace(0.01, 0.08, 8):
        penalty_u, penalty_force, penalty_iterations = solve_penalty(
            K, basis, top_y, anchor_x, bottom_y, gaps, imposed, penalty_u
        )
        active_u, multiplier, active_iterations = solve_active_set(
            K, basis, top_y, anchor_x, bottom_y, gaps, imposed, active_u
        )
        penalty_gap = penalty_u[bottom_y] + gaps
        active_gap = active_u[bottom_y] + gaps
        active_nodes = multiplier > 1.0e-9
        print(
            f"uy_top={imposed:7.4f} | penalty: contact={np.count_nonzero(penalty_force):2d}, "
            f"penetration={max(0.0, -penalty_gap.min()):.2e}, it={penalty_iterations:2d} | "
            f"PDAS: contact={np.count_nonzero(active_nodes):2d}, "
            f"min_gap={active_gap.min():+.2e}, force={multiplier.sum():.4f}, "
            f"it={active_iterations:2d}"
        )
        imposed_history.append(-imposed)
        force_history.append(multiplier.sum())

    assert active_gap.min() > -1.0e-10
    assert np.all(multiplier >= -1.0e-9)
    assert np.count_nonzero(active_nodes) > 1
    assert -penalty_gap.min() > 0.0

    nodal_u = active_u[basis.nodal_dofs]
    display_scale = 1.0
    deformed = mesh.p + display_scale * nodal_u
    triangulation = mtri.Triangulation(deformed[0], deformed[1], mesh.t.T)
    bottom_x = basis.doflocs[0, bottom_y]
    order = np.argsort(bottom_x)
    obstacle_y = -obstacle_gap(bottom_x)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    axes[0].triplot(triangulation, color="tab:blue", linewidth=0.5)
    axes[0].plot(bottom_x[order], obstacle_y[order], "k-", linewidth=2, label="rigid obstacle")
    axes[0].set(title=f"Deformation (×{display_scale:g})", xlabel="x", ylabel="y", aspect="equal")
    axes[0].legend()
    axes[1].plot(bottom_x, multiplier, "o-")
    axes[1].set(title="Final contact-force distribution", xlabel="x", ylabel="nodal contact force")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(imposed_history, force_history, "o-", label="contact force")
    axes[2].set(title="Contact response", xlabel="downward displacement", ylabel="total contact force")
    axes[2].grid(True, alpha=0.3)
    fig.savefig(Path(__file__).with_name("result.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
