from matplotlib import pyplot as plt


def plot_all_errors(rel_errors, errors):
    """
    Plot the relative errors and errors (Orth_A, Orth_B) of A and B in a grid layout
    """
    _, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Relative errors of A and B
    ax[0, 0].plot(
        range(1, len(rel_errors["A"]) + 1),
        rel_errors["A"],
        label="||A-A*||/||A*||",
        c="r",
    )
    ax[0, 0].set_xlabel("Iterations")
    ax[0, 0].set_ylabel("Relative Error")
    ax[0, 0].set_title("Convergence of `FactorizedGD (A)` to `A*`")
    ax[0, 0].legend()
    ax[0, 0].grid()

    # Relative errors of A and B
    ax[0, 1].plot(
        range(1, len(rel_errors["B"]) + 1),
        rel_errors["B"],
        label="||B-B*||/||B*||",
        c="g",
    )
    ax[0, 1].set_xlabel("Iterations")
    ax[0, 1].set_ylabel("Relative Error")
    ax[0, 1].set_title("Convergence of `FactorizedGD (B)` to `B*`")
    ax[0, 1].legend()
    ax[0, 1].grid()

    # Errors of A and B
    ax[1, 0].plot(
        range(1, len(errors["orth_A"]) + 1),
        errors["orth_A"],
        label="Orthogonal Loss A",
        c="b",
    )
    ax[1, 0].set_xlabel("Iterations")
    ax[1, 0].set_ylabel("Loss")
    ax[1, 0].set_title("Orthogonal Loss of `A`")
    ax[1, 0].legend()
    ax[1, 0].grid()

    # Errors of A and B
    ax[1, 1].plot(
        range(1, len(errors["orth_B"]) + 1),
        errors["orth_B"],
        label="Orthogonal Loss B",
        c="m",
    )
    ax[1, 1].set_xlabel("Iterations")
    ax[1, 1].set_ylabel("Loss")
    ax[1, 1].set_title("Orthogonal Loss of `B`")
    ax[1, 1].legend()
    ax[1, 1].grid()

    plt.tight_layout()
    plt.show()
