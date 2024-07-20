from dataclasses import dataclass


@dataclass
class GDArgs:
    d: int = 100  # Dimensionality of the input data
    r: int = 10  # Rank of the matrix W
    m: int = 50  # Number of columns in the matrix M
    n: int = 500  # Number of samples
    eta: float = 1e-5  # Learning rate
    eta_A: float = 1e-5  # Learning rate for A
    eta_B: float = 1e-5  # Learning rate for B
    iters: int = 1000  # Maximum number of iterations
    tau: int = 20  # Number of Orthogonalization steps per iteration
    epsilon: float = 1e-8  # Tolerance for Relative error (As compared to W*)
    lambda_orth: float = 0.7  # Regularization parameter for orthogonality
