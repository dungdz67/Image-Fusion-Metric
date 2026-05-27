def print_matrix_stats(matrix):
    import numpy as np
    matrix = np.asarray(matrix)
    flat = matrix.flatten()

    rows, cols = matrix.shape
    maximum = flat.max()
    minimum = flat.min()
    average = flat.mean()
    energy  = (flat ** 2).sum()

    print(f"Matrix Stats ({rows}x{cols})")
    print(f"  Min    : {minimum}")
    print(f"  Max    : {maximum}")
    print(f"  Avg    : {average:.4f}")
    print(f"  Energy : {energy}")
    print(f"  Count  : {flat.size}")