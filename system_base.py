################## Utility Functions ######################

def extract_partial_covmats(mat,corner):
    """Extract the partial covariance matrices from a gaussian mapping

    Args:
        mat (np.ndarray): Full covariance matrix
        corner (int): Index of the top left element of the bottom right matrix

    Returns:
        tuple: Matrices (A,B,D)
    """
    mat_a = mat[:corner, :corner]
    mat_b = mat[:corner, corner:]
    mat_d = mat[corner:, corner:]
    return mat_a, mat_b, mat_d