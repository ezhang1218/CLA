import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def cov_mle_centered(A):
    A = np.asarray(A)
    Ac = A - A.mean(axis=0, keepdims=True)
    return (Ac.T @ Ac) / A.shape[0]


def top_eigvec_sym(M):
    evals, evecs = np.linalg.eigh(M)
    return evecs[:, np.argmax(evals)]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def setup_axes():
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def add_legend(ax, cpca_label='CPCA'):
    legend_handles = [
        Line2D([0], [0], color='C0', lw=2, label='CLA'),
        Line2D([0], [0], color='C3', lw=2, label=cpca_label),
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        fontsize=8,
        frameon=True,
        borderpad=0.3,
        handlelength=1.5,
        labelspacing=0.3,
    )


def draw_vec_small(ax, v, label, style, offset=0.20, scale=1.12):
    v = v / np.linalg.norm(v)
    ax.arrow(0, 0, v[0], v[1], head_width=0.06, length_includes_head=True, **style)
    perp = np.array([-v[1], v[0]])
    perp /= np.linalg.norm(perp)
    ax.text(
        scale * v[0] + offset * perp[0],
        scale * v[1] + offset * perp[1],
        label, fontsize=8, ha='center', va='center',
    )


def draw_vec_large(ax, v, label, style, offset=0):
    v = v / np.linalg.norm(v)
    ax.arrow(0, 0, v[0], v[1], head_width=0.06, length_includes_head=True, **style)
    perp = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    ax.text(
        1.08 * v[0] + offset * perp[0],
        1.08 * v[1] + offset * perp[1],
        label, fontsize=11, ha='center', va='center',
    )


# ---------------------------------------------------------------------------
# Simulation 1: Variance increases with dose (equal sample sizes)
# ---------------------------------------------------------------------------

def simulation_variance():
    np.random.seed(8)

    p = 2
    sigma_eps = 1
    n0 = 1000
    z_levels = np.array([1.0, 1.5, 2.0])

    u_map = {
        1.0: np.array([np.cos(np.deg2rad(-40)), np.sin(np.deg2rad(-40))]),
        1.5: np.array([np.cos(np.deg2rad(20)),  np.sin(np.deg2rad(20))]),
        2.0: np.array([np.cos(np.deg2rad(70)),  np.sin(np.deg2rad(70))]),
    }
    s = np.array([1.0, 0.0])
    n_per_z = {1.0: 1000, 1.5: 1000, 2.0: 1000}
    a = lambda z: np.sqrt(z)

    X_list, Z_list = [], []

    xi0 = np.random.standard_normal((n0, 1))
    E0  = np.random.normal(0, sigma_eps, size=(n0, p))
    Y   = xi0 @ s[None, :] + E0
    X_list.append(Y)
    Z_list.append(np.zeros(n0))

    for z in z_levels:
        n   = n_per_z[z]
        xi  = np.random.standard_normal((n, 1))
        eta = np.random.standard_normal((n, 1))
        Ez  = np.random.normal(0, sigma_eps, size=(n, p))
        Xt  = xi @ s[None, :] + (a(z) * eta) @ u_map[z][None, :] + Ez
        X_list.append(Xt)
        Z_list.append(np.full(n, z))

    X = np.vstack(X_list)
    Z = np.concatenate(Z_list)

    Sigma0 = cov_mle_centered(X[Z == 0])
    Sz = {z: cov_mle_centered(X[Z == z]) for z in z_levels}

    T_cla  = sum((Sz[z] - Sigma0) / z for z in z_levels) / len(z_levels)
    v_cla  = top_eigvec_sym(T_cla)

    T_cpca = cov_mle_centered(X[Z > 0]) - Sigma0
    v_cpca = top_eigvec_sym(T_cpca)

    fig, ax = setup_axes()
    draw_vec_small(ax, u_map[1.0], "(z=1.0)", dict(color='gray'), offset=0.20)
    draw_vec_small(ax, u_map[1.5], "(z=1.5)", dict(color='gray'), offset=0.20)
    draw_vec_small(ax, u_map[2.0], "(z=2.0)", dict(color='gray'), offset=0.15)
    ax.arrow(0, 0, -v_cla[0],  -v_cla[1],  head_width=0.06, color='C0', lw=2)
    ax.arrow(0, 0, -v_cpca[0], -v_cpca[1], head_width=0.06, color='C3', lw=2)
    add_legend(ax)
    plt.tight_layout(pad=0.3)
    # fig.savefig("simulation_variance.png", dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Simulation 2: Unequal sample sizes, constant effect amplitude
# ---------------------------------------------------------------------------

def simulation_sample_size():
    p = 2
    sigma = 1
    n0 = 1000
    z_levels = np.array([1.0, 1.5, 2.0])

    u_map = {
        1.0: np.array([np.cos(np.deg2rad(-40)), np.sin(np.deg2rad(-40))]),
        1.5: np.array([np.cos(np.deg2rad(20)),  np.sin(np.deg2rad(20))]),
        2.0: np.array([np.cos(np.deg2rad(70)),  np.sin(np.deg2rad(70))]),
    }
    s = np.array([1.0, 0.0])
    n_per_z = {1.0: 1000, 1.5: 1000, 2.0: 10000}
    a = lambda z: 1.0

    X_list, Z_list = [], []

    xi0 = np.random.standard_normal((n0, 1))
    E0  = np.random.normal(0, sigma, size=(n0, p))
    Y   = xi0 @ s[None, :] + E0
    X_list.append(Y)
    Z_list.append(np.zeros(n0))

    for z in z_levels:
        n   = n_per_z[z]
        xi  = np.random.standard_normal((n, 1))
        eta = np.random.standard_normal(n)[:, None]
        Ez  = np.random.normal(0, sigma, size=(n, p))
        Xt  = xi @ s[None, :] + (a(z) * eta) @ u_map[z][None, :] + Ez
        X_list.append(Xt)
        Z_list.append(np.full(n, z))

    X = np.vstack(X_list)
    Z = np.concatenate(Z_list)

    Sigma0 = cov_mle_centered(X[Z == 0])
    Sz = {z: cov_mle_centered(X[Z == z]) for z in z_levels}

    T_cla  = sum((Sz[z] - Sigma0) / z for z in z_levels) / len(z_levels)
    v_cla  = top_eigvec_sym(T_cla)

    T_cpca = cov_mle_centered(X[Z > 0]) - Sigma0
    v_cpca = top_eigvec_sym(T_cpca)

    fig, ax = setup_axes()
    draw_vec_small(ax, u_map[1.0], "(z=1.0)", dict(color='gray'), offset=0.20)
    draw_vec_small(ax, u_map[1.5], "(z=1.5)", dict(color='gray'), offset=0.20)
    draw_vec_small(ax, u_map[2.0], "(z=2.0)", dict(color='gray'), offset=0.15)
    ax.arrow(0, 0, -v_cla[0],  -v_cla[1], head_width=0.06, color='C0', lw=2)
    ax.arrow(0, 0,  v_cpca[0],  v_cpca[1], head_width=0.06, color='C3', lw=2)
    add_legend(ax)
    plt.tight_layout(pad=0.3)
    fig.savefig("simulation_sample.png", dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Simulation 3: Amplitude scales linearly with dose
# ---------------------------------------------------------------------------

def simulation_linear_amplitude():
    p = 2
    sigma = 0.1
    n0 = 1000
    z_levels = np.array([1.0, 2.0, 3.0])

    u_map = {
        1.0: np.array([np.cos(np.deg2rad(-20)), np.sin(np.deg2rad(-20))]),
        2.0: np.array([np.cos(np.deg2rad(40)),  np.sin(np.deg2rad(40))]),
        3.0: np.array([np.cos(np.deg2rad(70)),  np.sin(np.deg2rad(70))]),
    }
    s = np.array([1.0, 1.0])
    n_per_z = {1.0: 1000, 2.0: 1000, 3.0: 1000}
    a = lambda z: z

    X_list, Z_list = [], []

    xi0 = np.random.standard_normal((n0, 1))
    E0  = np.random.normal(0, sigma, size=(n0, p))
    Y   = xi0 @ s[None, :] + E0
    X_list.append(Y)
    Z_list.append(np.zeros(n0))

    for z in z_levels:
        n   = n_per_z[z]
        xi  = np.random.standard_normal((n, 1))
        eta = np.random.standard_normal(n)[:, None]
        Ez  = np.random.normal(0, sigma, size=(n, p))
        Xt  = xi @ s[None, :] + (a(z) * eta) @ u_map[z][None, :] + Ez
        X_list.append(Xt)
        Z_list.append(np.full(n, z))

    X = np.vstack(X_list)
    Z = np.concatenate(Z_list)

    Sigma0 = cov_mle_centered(X[Z == 0])
    Sz = {z: cov_mle_centered(X[Z == z]) for z in z_levels}

    T_cla  = sum((Sz[z] - Sigma0) / z for z in z_levels) / len(z_levels)
    v_cla  = top_eigvec_sym(T_cla)

    T_cpca = cov_mle_centered(X[Z > 0]) - Sigma0
    v_cpca = top_eigvec_sym(T_cpca)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])

    draw_vec_large(ax, u_map[1.0], "(z=1.0)", dict(color='gray'), offset=0.20)
    draw_vec_large(ax, u_map[2.0], "(z=1.5)", dict(color='gray'), offset=0.20)
    draw_vec_large(ax, u_map[3.0], "(z=2.0)", dict(color='gray'), offset=0.15)

    ax.arrow(0, 0,  v_cpca[0],  v_cpca[1], head_width=0.06, color='C3', lw=2, length_includes_head=True)
    ax.arrow(0, 0, -v_cla[0],  -v_cla[1],  head_width=0.06, color='C0', lw=2, length_includes_head=True)

    legend_handles = [
        Line2D([0], [0], color='C0', lw=2, label='CLA'),
        Line2D([0], [0], color='C3', lw=2, label='CPCA (pooled)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Simulation 4: High variance at z=1, dose-squared weighting in CLA
# ---------------------------------------------------------------------------

def simulation_high_variance_z1():
    p = 2
    sigma = 1
    n0 = 1000
    z_levels = np.array([1.0, 2.0, 3.0])

    u_map = {
        1.0: np.array([np.cos(np.deg2rad(-20)), np.sin(np.deg2rad(-20))]),
        2.0: np.array([np.cos(np.deg2rad(40)),  np.sin(np.deg2rad(40))]),
        3.0: np.array([np.cos(np.deg2rad(70)),  np.sin(np.deg2rad(70))]),
    }
    s = np.array([1.0, 1.0])
    n_per_z = {1.0: 1000, 2.0: 1000, 3.0: 1000}
    a = lambda z: np.sqrt(z) * (2.0 if z == 1.0 else 1.0)

    X_list, Z_list = [], []

    xi0 = np.random.standard_normal((n0, 1))
    E0  = np.random.normal(0, sigma, size=(n0, p))
    Y   = xi0 @ s[None, :] + E0
    X_list.append(Y)
    Z_list.append(np.zeros(n0))

    for z in z_levels:
        n   = n_per_z[z]
        xi  = np.random.standard_normal((n, 1))
        eta = np.random.standard_normal((n, 1))
        Ez  = np.random.normal(0, sigma, size=(n, p))
        Xt  = xi @ s[None, :] + (a(z) * eta) @ u_map[z][None, :] + Ez
        X_list.append(Xt)
        Z_list.append(np.full(n, z))

    X = np.vstack(X_list)
    Z = np.concatenate(Z_list)

    Sigma0 = cov_mle_centered(X[Z == 0])
    Sz = {z: cov_mle_centered(X[Z == z]) for z in z_levels}

    T_cla  = sum((Sz[z] - Sigma0) / z**2 for z in z_levels) / len(z_levels)
    v_cla  = top_eigvec_sym(T_cla)

    T_cpca = cov_mle_centered(X[Z > 0]) - Sigma0
    v_cpca = top_eigvec_sym(T_cpca)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])

    draw_vec_large(ax, u_map[1.0], "(z=1)", dict(color='gray'), offset=0.15)
    draw_vec_large(ax, u_map[2.0], "(z=2)", dict(color='gray'))
    draw_vec_large(ax, u_map[3.0], "(z=3)", dict(color='gray'))

    ax.arrow(0, 0,  v_cpca[0],  v_cpca[1], head_width=0.06, color='C3', lw=2, length_includes_head=True)
    ax.arrow(0, 0, -v_cla[0],  -v_cla[1],  head_width=0.06, color='C0', lw=2, length_includes_head=True)

    legend_handles = [
        Line2D([0], [0], color='C0', lw=2, label='CLA'),
        Line2D([0], [0], color='C3', lw=2, label='CPCA (pooled)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=11)
    plt.tight_layout()
    # fig.savefig("simulation_highvar_z1.png", dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    simulation_variance()
    simulation_sample_size()
    simulation_linear_amplitude()
    simulation_high_variance_z1()
