import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _second_moment(X_group: np.ndarray, center: bool = True) -> np.ndarray:
    """
    Compute the second-moment (covariance-like) matrix for a group.

    Parameters
    ----------
    X_group : ndarray of shape (n, p)
    center  : if True, mean-centre the group before computing X^T X / n

    Returns
    -------
    ndarray of shape (p, p)
    """
    n_g = X_group.shape[0]
    if n_g == 0:
        raise ValueError("Encountered an empty group.")
    if center:
        Xc = X_group - X_group.mean(axis=0, keepdims=True)
        return (Xc.T @ Xc) / n_g
    else:
        return X_group.T @ X_group


def _sorted_eigh(M: np.ndarray):
    """
    Eigen-decomposition of a symmetric matrix, eigenvalues sorted descending.

    Returns
    -------
    evals : ndarray (p,)
    evecs : ndarray (p, p)  – columns are eigenvectors
    """
    evals, evecs = np.linalg.eigh(M)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def cla(
    X: np.ndarray,
    Z: np.ndarray,
    center: bool = True,
    beta: float = 1.0,
):
    """
    Contrastive Liquid Association (CLA).

    Computes a matrix T whose leading eigenvectors capture the directions in
    feature space along which treated groups differ from control, weighted by
    dose / treatment level.

    Parameters
    ----------
    X      : ndarray of shape (n, p) – feature matrix
    Z      : ndarray of shape (n,)   – treatment assignment
                 0  → control
                 >0 → treated (can be multi-level / continuous dose)
    center : bool – whether to mean-centre within each group (default True)
    beta   : float – reserved scaling parameter (default 1.0, unused in T)

    Returns
    -------
    T      : ndarray (p, p) – the CLA matrix
    evals  : ndarray (p,)   – eigenvalues, descending
    evecs  : ndarray (p, p) – eigenvectors as columns (evecs[:, k] is the k-th)
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)

    ctrl = Z == 0
    trt  = Z > 0

    if ctrl.sum() == 0:
        raise ValueError("No control observations found (Z == 0).")
    if trt.sum() == 0:
        raise ValueError("No treated observations found (Z > 0).")

    Sigma0    = _second_moment(X[ctrl], center=center)
    X_treated = X[trt]
    z_treated = Z[trt]
    n_plus    = float(trt.sum())

    p = X.shape[1]
    T = np.zeros((p, p), dtype=float)

    for z_val in np.unique(z_treated):
        idx    = z_treated == z_val
        n_z    = float(idx.sum())
        Sigma_z = _second_moment(X_treated[idx], center=center)
        T += (Sigma_z - Sigma0) / z_val

    T /= n_plus

    evals, evecs = _sorted_eigh(T)
    return T, evals, evecs


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def plot_cla_eigenvectors(
    evecs: np.ndarray,
    feature_names,
    n_components: int = 2,
    figsize: tuple = (7, 5),
):
    """
    Bar-image plot of the leading CLA eigenvectors.

    Parameters
    ----------
    evecs         : ndarray (p, p) – output of cla()
    feature_names : array-like of length p
    n_components  : how many leading eigenvectors to plot (1 or 2)
    figsize       : figure size passed to plt.subplots
    """
    n_components = min(n_components, evecs.shape[1])
    vecs   = [evecs[:, k] for k in range(n_components)]
    titles = [f"CLA eigenvector {k+1}" for k in range(n_components)]

    vabs = max(np.abs(v).max() for v in vecs)
    vmin, vmax = -vabs, vabs

    fig, axes = plt.subplots(1, n_components, figsize=figsize,
                             constrained_layout=True)
    if n_components == 1:
        axes = [axes]

    for k, (ax, vec, title) in enumerate(zip(axes, vecs, titles)):
        ax.imshow(vec.reshape(-1, 1), aspect="auto",
                  cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Eigenvector", fontsize=14)
        ax.set_xticks([0])
        ax.set_xticklabels(["0"], fontsize=12)
        ax.set_yticks(np.arange(len(feature_names)))
        if k == 0:
            ax.set_yticklabels(feature_names, fontsize=12)
        else:
            ax.set_yticklabels([])

    cbar = fig.colorbar(axes[-1].images[0], ax=axes, shrink=0.8)
    cbar.set_label("Loading", fontsize=14)
    plt.show()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, p = 300, 10
    feature_names = [f"feature_{i}" for i in range(p)]

    # Simulate data: control (Z=0), low dose (Z=1), high dose (Z=2)
    X = rng.standard_normal((n, p))
    Z = rng.choice([0, 1, 2], size=n, p=[0.4, 0.3, 0.3])

    # Inject a signal into the first two features for treated units
    X[Z > 0, 0] += Z[Z > 0] * 1.5
    X[Z > 0, 1] -= Z[Z > 0] * 0.8

    T, evals, evecs = cla(X, Z, center=True)

    print("Top eigenvalues:", evals[:5])
    print("First eigenvector:\n", evecs[:, 0])

    plot_cla_eigenvectors(evecs, feature_names, n_components=2)
