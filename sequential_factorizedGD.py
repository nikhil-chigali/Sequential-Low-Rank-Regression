import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from loguru import logger
from functools import partial


def setup(d: int, n: int, m: int, r: int) -> tuple:
    # Generate the data
    M = np.random.randn(m, d)
    U, _, Vt = la.svd(M, full_matrices=False)

    # Strictly decaying eigenvalues
    S = np.linspace(10, 1, num=r)
    S_root = np.diag(np.sqrt(S))  # r x r
    logger.debug(f"Eigenvalues: {S}")

    # Generate the factors
    A_star_hat = U[:, :r]
    B_star_hat = Vt[:r, :]

    A_star = A_star_hat @ S_root  # m x r
    B_star = S_root @ B_star_hat  # r x d

    # Generate the observations
    X = np.random.randn(d, n)  # d x n
    Y = A_star @ B_star @ X  # m x n

    W_star = A_star @ B_star  # m x d

    logger.debug(f"Rank of W_star: {la.matrix_rank(W_star)}")

    return X, Y, W_star, A_star, B_star


def MSE_loss_factorized(X, Y, A, B, grad=False):
    # Depends on the data X and Y
    if grad:
        grad_A = (A @ B @ X - Y) @ X.T @ B.T
        grad_B = A.T @ (A @ B @ X - Y) @ X.T

        return grad_A, grad_B
    return 0.5 * la.norm(Y - A @ B @ X, ord="fro") ** 2


# Gradient Descent Update Rule
def MSE_GD_Update(X, Y, A_old, B_old, eta_A, eta_B):
    """
    Update the factorized matrices A and B using the gradient descent update rule
    """
    grad_A, grad_B = MSE_loss_factorized(X, Y, A_old, B_old, grad=True)
    A_new = A_old - eta_A * grad_A
    B_new = B_old - eta_B * grad_B
    return A_new, B_new


def GS_Orthogonalization(A_prev, B_prev, A_k, B_k):
    # A_prev: m x k-1 | B_prev: k-1 x d
    # A_k: m x 1 | B_k: 1 x d

    # Subtract the projection of A_k onto the previous factors
    A_k = A_k - np.sum(
        (A_k.T @ A_prev) * A_prev / np.sum(A_prev**2, axis=0, keepdims=True),
        axis=1,
        keepdims=True,
    )

    # Subtract the projection of B_k onto the previous factors
    B_k = B_k - np.sum(
        (B_prev @ B_k.T) * B_prev / np.sum(B_prev**2, axis=1, keepdims=True),
        axis=0,
        keepdims=True,
    )

    return A_k, B_k


def rank1_factorizedGD(
    X,
    Y,
    W_star,
    eta_A,
    eta_B,
    projection=None,
    max_iter=500,
):
    d = X.shape[0]
    m = Y.shape[0]

    # Initialize the factors
    A = np.random.randn(m, 1)
    B = np.random.randn(1, d)

    # Normalize the factors
    A = A / la.norm(A)
    B = B / la.norm(B)

    # Initialize the errors
    rel_error_W = []
    mse = []

    for _ in range(max_iter):

        # GD Update
        A, B = MSE_GD_Update(X, Y, A, B, eta_A, eta_B)

        # Projection
        if projection:
            A, B = projection(A, B)

        # Compute the errors
        mse.append(MSE_loss_factorized(X, Y, A, B))

        W = A @ B
        rel_error_W.append(la.norm(W - W_star, "fro") / la.norm(W_star, "fro"))

    return A, B, {"rel_error_W": rel_error_W, "mse": mse}


def plot_figures(X, Y, A_star, B_star, A_r, B_r, error_logs, args):
    folder = f"results/exps/sequential_factorizedGD_d={args.d}_n={args.n}_m={args.m}_r={args.r}_orthogonalize={args.orthogonalize}"

    logger.debug("Plotting the Training Results")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for k in range(args.r):
        ax[0].plot(error_logs[k]["rel_error_W"], label=f"Rank-{k+1}")
        ax[1].plot(error_logs[k]["mse"], label=f"Rank-{k+1}")

    ax[0].set_title("Relative Error in (A_kB_k, W*)")
    ax[1].set_title("MSE")
    ax[0].set_xlabel("Iterations")
    ax[1].set_xlabel("Iterations")
    ax[0].set_ylabel("Relative Error")
    ax[1].set_ylabel("MSE")
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "training_results.png"))
    plt.show()

    logger.debug("Plotting the Factors")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(A_star @ B_star, cmap="viridis")
    ax[0].set_title("A* @ B*")
    ax[1].imshow(A_r @ B_r, cmap="viridis")
    ax[1].set_title("A @ B")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "A_star_B_star_vs_A_r_B_r.png"))
    plt.show()

    ranks = range(1, args.r + 1)
    logger.debug("Plotting final errors of each component of A and B")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(
        ranks, [error_logs[k]["rel_error_W"][-1] for k in range(args.r)], marker="o"
    )
    ax[1].plot(ranks, [error_logs[k]["mse"][-1] for k in range(args.r)], marker="o")
    ax[0].set_title("Relative Error in (A_kB_k, W*)")
    ax[1].set_title("MSE")
    ax[0].set_xticks(ranks)
    ax[1].set_xticks(ranks)
    ax[0].set_xlabel("Rank")
    ax[1].set_xlabel("Rank")
    ax[0].set_ylabel("Relative Error")
    ax[1].set_ylabel("MSE")
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "final_errors_A_kB_k_W_star_vs_A_rB_r.png"))
    plt.show()

    logger.debug("Plotting cumulative errors of each component of A and B")
    rel_errors = [
        la.norm(A_r[:, : k + 1] @ B_r[: k + 1, :] - A_star @ B_star, "fro")
        / la.norm(A_star @ B_star, "fro")
        for k in range(args.r)
    ]
    mse = [
        MSE_loss_factorized(X, Y, A_r[:, : k + 1], B_r[: k + 1, :])
        for k in range(args.r)
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(ranks, rel_errors, marker="o")
    ax[1].plot(ranks, mse, marker="o")
    ax[0].set_xticks(ranks)
    ax[1].set_xticks(ranks)
    ax[0].set_title("Relative Error - Cumulative (A_rB_r, W*)")
    ax[1].set_title("MSE - Cumulative A_rB_r")
    ax[0].set_xlabel("Rank-r")
    ax[1].set_xlabel("Rank-r")
    ax[0].set_ylabel("Relative Error")
    ax[1].set_ylabel("MSE")
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "cumulative_errors_A_rB_r_W_star_vs_A_rB_r.png"))
    plt.show()


def main(args):
    logger.info(args)
    logger.info("Setting up the data")
    X, Y, W_star, A_star, B_star = setup(args.d, args.n, args.m, args.r)

    A_r = np.zeros_like(A_star)
    B_r = np.zeros_like(B_star)

    Y_k = Y
    error_logs = []
    projection = None

    logger.info("Running the sequential factorized GD")
    for k in range(args.r):
        logger.info(f"Retrieving rank-{k+1} component")

        # Orthogonalization
        if args.orthogonalize:
            projection = partial(GS_Orthogonalization, A_r[:, :k], B_r[:k, :])

        # Run the factorized GD
        A_k, B_k, logs = rank1_factorizedGD(
            X, Y_k, W_star, args.eta_A, args.eta_B, projection, args.max_iter
        )

        # Store the results
        A_r[:, k] = A_k.ravel()
        B_r[k, :] = B_k.ravel()

        error_logs.append(logs)

        # Deflate the components
        Y_k = Y_k - A_k @ B_k @ X

    # Save the results
    folder = f"results/exps/sequential_factorizedGD_d={args.d}_n={args.n}_m={args.m}_r={args.r}_orthogonalize={args.orthogonalize}"
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, "results.npz")
    logger.info(f"Saving the results to {fname}")
    np.savez(fname, {"A": A_r, "B": B_r, "error_logs": error_logs})

    fname = os.path.join(folder, "data.npz")
    logger.info(f"Saving X, Y, A*, B* to {fname}")
    np.savez(fname, {"A_star": A_star, "B_star": B_star, "X": X, "Y": Y})

    # Plot and save the results
    plot_figures(X, Y, A_star, B_star, A_r, B_r, error_logs, args)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--d", type=int, default=100, help="Number of features")
    argparser.add_argument("--n", type=int, default=100, help="Number of samples")
    argparser.add_argument("--m", type=int, default=100, help="Number of observations")
    argparser.add_argument("--r", type=int, default=10, help="Rank of the factors")
    argparser.add_argument(
        "--orthogonalize", action="store_true", help="Orthogonalize the factors"
    )
    argparser.add_argument(
        "--eta_A", type=float, default=0.0001, help="Learning rate for A"
    )
    argparser.add_argument(
        "--eta_B", type=float, default=0.0001, help="Learning rate for B"
    )
    argparser.add_argument(
        "--max_iter", type=int, default=500, help="Maximum iterations"
    )

    args = argparser.parse_args()
    main(args)
