import numpy as np


def grassman_distance(A, B):
    """
    Grassman distance between subspaces of A and B.
    B: d x r matrix
    A: r x k matrix
    """
    Ua, Sa, Va = np.linalg.svd(A)
    Ub, Sb, Vb = np.linalg.svd(B)

    print(Ua.shape, Sa.shape, Va.shape)
    print(Ub.shape, Sb.shape, Vb.shape)
