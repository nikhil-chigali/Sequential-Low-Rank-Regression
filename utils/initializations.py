import numpy as np
import numpy.linalg as la

from args import GDArgs


def random_AB(args: GDArgs):
    """
    Random initialization of A and B
    """
    A = np.random.randn(args.d, args.r)
    B = np.random.randn(args.m, args.r)

    # Normalizing columns of A and B
    A = A / la.norm(A, axis=0, keepdims=True)
    B = B / la.norm(B, axis=0, keepdims=True)

    return A, B


def orthonormal_AB(args: GDArgs):
    """
    Orthonormal initialization of A and B
    """
    A = np.random.randn(args.d, args.r)
    B = np.random.randn(args.m, args.r)

    # Orthogonalizing A and B
    A, _ = la.qr(A)
    B, _ = la.qr(B)

    return A, B
