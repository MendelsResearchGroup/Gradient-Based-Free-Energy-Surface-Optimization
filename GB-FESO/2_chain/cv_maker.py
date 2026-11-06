"""Pair-distance RMSDs with independent chain reversals and no chain exchange."""

import numpy as np
import torch


# Geometric 2D templates in atom-ID order, unwrapped across periodic boundaries.
RMSD_REFERENCE_NAMES = ("extended_interface", "end_anchored_dimer")
RMSD_REFERENCE_COORDINATES = np.array([
    # Original train/0053/dump.lammpstrj, timestep 2306400.
    [
        [-2.71480,  0.06354],
        [-1.29580,  0.44754],
        [ 0.13720,  0.94354],
        [ 1.66420,  1.28754],
        [ 3.20920,  1.31854],
        [ 2.72720, -0.17146],
        [ 1.22320, -0.39446],
        [-0.21780, -0.69546],
        [-1.67580, -1.11446],
        [-3.05680, -1.68486],
    ],
    # Original train/0237/dump.lammpstrj, timestep 7181400.
    [
        [-0.92800, -2.40637],
        [ 0.62400, -2.30127],
        [ 1.51100, -1.10247],
        [ 1.46000,  0.37973],
        [-0.03600,  0.73473],
        [-0.12900, -0.93677],
        [-1.39200, -0.11657],
        [-1.63600,  1.39993],
        [-0.49100,  2.33683],
        [ 1.01700,  2.01223],
    ],
])
RMSD_REFERENCE_DISTANCES = np.linalg.norm(
    RMSD_REFERENCE_COORDINATES[:, :, None, :]
    - RMSD_REFERENCE_COORDINATES[:, None, :, :],
    axis=-1,
)

# Identity, reverse chain 1, reverse chain 2, reverse both. No chain exchange.
CHAIN_SYMMETRY_PERMUTATIONS = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 3, 2, 1, 0, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 9, 8, 7, 6, 5],
    [4, 3, 2, 1, 0, 9, 8, 7, 6, 5],
])


def rmsd_cvs(distance_matrices, reference_distances=RMSD_REFERENCE_DISTANCES):
    """Return (N, R) distance RMSDs to R reference distance matrices.

    Input is (N, 10, 10), ordered by atom IDs 1..10. For each reference,
    return min_p sqrt(mean_{i<j}((d_ij - reference[p_i, p_j])**2)), using
    all 45 unique pairs equally and the four chain reversals above.
    Matrices are assumed symmetric; the diagonal and lower half are unused.

    Values retain length units. Tensors retain gradients; other inputs return
    NumPy arrays. The minimum is piecewise differentiable, with a finite zero
    subgradient at an exact match.
    """
    distances = distance_matrices
    if not isinstance(distances, torch.Tensor):
        distances = torch.from_numpy(np.array(distances, dtype=float))
    elif not distances.is_floating_point():
        distances = distances.float()
    if distances.ndim != 3 or distances.shape[1:] != (10, 10):
        raise ValueError("Expected distance matrices with shape (N, 10, 10).")

    references = torch.as_tensor(
        reference_distances, dtype=distances.dtype, device=distances.device,
    )
    if references.ndim != 3 or references.shape[1:] != (10, 10) or len(references) == 0:
        raise ValueError("Expected reference distance matrices with shape (R, 10, 10).")
    permutations = torch.as_tensor(
        CHAIN_SYMMETRY_PERMUTATIONS, device=distances.device,
    )
    rows, columns = torch.triu_indices(10, 10, offset=1, device=distances.device)
    pairs = distances[:, rows, columns]
    variants = []
    for permutation in permutations:
        reference_pairs = references[:, permutation[rows], permutation[columns]]
        difference = pairs[:, None, :] - reference_pairs[None, :, :]
        variants.append(torch.linalg.vector_norm(difference, dim=-1) / np.sqrt(45))
    result = torch.stack(variants).amin(dim=0)
    return result if isinstance(distance_matrices, torch.Tensor) else result.numpy()
