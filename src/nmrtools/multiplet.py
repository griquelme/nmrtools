import numpy as np
from itertools import product
from math import log2, log10
from .utils import lorentz
from sklearn.utils.extmath import cartesian
from typing import List, Optional, Tuple


def annotate_multiplet(f: np.ndarray, height: np.ndarray) -> List[List[float]]:
    max_delta = 4
    candidates = _find_candidate_heights(height, max_delta)
    valid_annotations = list()
    for c_height in candidates:
        j_list = _multiplet_to_j_list(f, c_height)
        if j_list is not None:
            valid_annotations.append(j_list)
    return valid_annotations


def _is_symmetric(height: np.ndarray) -> bool:
    """
    Check if the heights of a multiplet are symmetric.

    Parameters
    ----------
    height : array

    Returns
    -------
    bool

    """
    q, r = divmod(height.size, 2)
    left = height[:q]
    right = height[height.size:q - 1 + r:-1]
    return np.array_equal(left, right)


def _find_candidate_heights(height, max_delta: int) -> List[np.ndarray]:
    """

    Adds perturbations to multiplets candidates to meet symmetry and intensity
    ratio requirements

    Parameters
    ----------
    height : array
        height of each peak in the multiplet.
    max_delta : int
        Maximum perturbation to add to the candidates.

    Returns
    -------
    candidates : List[np.ndarray]
        List of candidate heights

    """
    # normalize min intensity to 1, round intensity to integers
    height = np.round(height / min(height)).astype(int)

    # check if total intensity is a power of 2
    # if not, add perturbations until total int is a power of 2
    n = height.sum()
    closest_power_of_2 = 2 ** round(log2(n))
    delta = n - closest_power_of_2

    if abs(delta) <= max_delta:
        perturbations = cartesian([[-1, 0, 1] for _ in range(height.size)])
        perturbations_sum = perturbations.sum(axis=1)
        perturbations_abs_sum = np.abs(perturbations).sum(axis=1)
        valid_mask = (
            (perturbations_sum == delta) & (perturbations_abs_sum <= max_delta)
        )
        perturbations = perturbations[valid_mask, :]
        candidates = list(perturbations + height)
    elif n == closest_power_of_2:
        candidates = [height]
    else:
        candidates = list()

    # check symmetric multiplet
    candidates = [x for x in candidates if _is_symmetric(x)]

    # check minimums at the extremes
    candidates = [x for x in candidates if ((x[0] == 1) and (x[-1] == 1))]

    return candidates


def _j_list_to_height(
        j_list: List[float], j_tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a list of coupling constants into a vector of peak frequencies and
    heights.

    Parameters
    ----------
    j_list : list[float]
    j_tol : float
        Tolerance to merge close frequency values.

    Returns
    -------
    freq : array
        frequencies of each peak in the multiplet
    height : array
        Height og each peak in the multiplet.
    """

    # find all frequency values
    combinations = cartesian([[-0.5, 0.5] for _ in range(len(j_list))])
    # Rounds to at least the same number of decimals than j_tol
    n_round = max(1, -round(log10(j_tol)) + 1)
    peak_combs = (combinations * j_list).sum(axis=1).round(n_round)
    peak, height = np.unique(peak_combs, return_counts=True)

    # merge close peaks
    merge_index = np.where(np.diff(peak) < j_tol)[0]
    while merge_index.size:
        x0 = peak[merge_index]
        x1 = peak[merge_index + 1]
        y0 = height[merge_index]
        y1 = height[merge_index + 1]
        y_sum = y0 + y1
        # merged freq is the weighted average of the merged peaks
        # merged height is the total height of each peak
        peak[merge_index + 1] = (x0 * y0 + x1 * y0) / y_sum
        height[merge_index + 1] = y_sum
        np.delete(peak, merge_index)
        np.delete(height, merge_index)
        merge_index = np.where(np.diff(peak) < j_tol)[0]
    return peak, height


def _multiplet_to_j_list(
    f: np.ndarray, height: np.ndarray) -> Optional[List[float]]:
    return [1.0]


def simulate_multiplet_frequency(
    f: np.ndarray, j: list, delta: float, t2: float, normalize=True
):
    result = np.zeros_like(f)
    j = np.array(j)
    for p in product([-0.5, 0.5], repeat=j.size):
        mult_omega = (j * p).sum() + delta
        result += lorentz(f, mult_omega, t2)
    if normalize:
        result /= max(result)
    return result
