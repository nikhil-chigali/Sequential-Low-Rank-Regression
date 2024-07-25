from matplotlib import pyplot as plt


def plot_all_errors(exp_name: str, errors: dict) -> None:
    """
    Plot the relative error, MSE and Orthogonalization error (Orth_A, Orth_B) of A and B in a grid layout

    Parameters:
    exp_name (str): Name of the experiment
    errors (dict): Dictionary containing the errors of the model

    Returns:
    None
    """
    # Check if the orthogonalization errors are present
    plot_orth = "orth_A" in errors and "orth_B" in errors

    if plot_orth:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        ax1, ax2 = axs[0]
    else:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        ax1, ax2 = axs

    fig.suptitle(exp_name)

    # Plot the relative error of W
    ax1.plot(errors["rel_error_W"], label="Relative Error of W")
    ax1.set_ylim(top=1, bottom=0)
    ax1.set_title("Relative Error of (W,W*)")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Relative Error")
    ax1.grid()
    ax1.legend()

    # Plot the MSE loss
    ax2.plot(errors["mse"], label="MSE Loss")
    # ax2.set_ylim(top=5000, bottom=0)
    ax2.set_title("MSE Loss")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("MSE Loss")
    ax2.grid()
    ax2.legend()

    if plot_orth:
        # Plot the orthogonalization errors
        axs[1, 0].plot(errors["orth_A"], label="Orthogonal Constraint Loss A")
        axs[1, 0].set_title("Orthogonal Constraint Loss A")
        axs[1, 0].set_xlabel("Iterations")
        axs[1, 0].set_ylabel("Orthogonal Constraint Loss")
        axs[1, 0].grid()
        axs[1, 0].legend()

        axs[1, 1].plot(errors["orth_B"], label="Orthogonal Constraint Loss B")
        axs[1, 1].set_title("Orthogonal Constraint Loss B")
        axs[1, 1].set_xlabel("Iterations")
        axs[1, 1].set_ylabel("Orthogonal Constraint Loss")
        axs[1, 1].grid()
        axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
