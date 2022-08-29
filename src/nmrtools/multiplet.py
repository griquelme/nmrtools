import numpy as np
from itertools import permutations
from math import log2
from .utils import fid, find_closest, lorentz
from sklearn.utils.extmath import cartesian
from typing import List, Optional, Tuple


def simulate_frequency(
    f: np.ndarray, j_list: List[float], t2: float, f0: float = 0.0
) -> np.ndarray:
    result = np.zeros_like(f)
    j_tol = f[1] - f[0]
    f_peaks, h_peaks = _j_list_to_height(j_list, j_tol)
    for fp, hp in zip(f_peaks, h_peaks):
        result += hp * lorentz(f, fp + f0, t2)
    return result


def simulate_time(
    t: np.ndarray, j_list: List[float], t2: float, f0: float = 0, phase: float = 0.0
) -> np.ndarray:
    result = np.zeros(t.size, dtype="complex128")
    j_tol = 1 / t[-1]
    f_peaks, h_peaks = _j_list_to_height(j_list, j_tol)
    for fp, hp in zip(f_peaks, h_peaks):
        result += hp * fid(t, t2, fp + f0, phase)
    return result


def annotate(
    f: np.ndarray, height: np.ndarray, f_tol: float, max_perturbation: int = 1
) -> List[List[float]]:
    """
    Finds the couplings constants for a multiplet.


    Parameters
    ----------
    f : np.ndarray
        sorted frequency of the multiplet peaks.
    height : np.ndarray
        Intensity of the peaks.
    f_tol : float
        If two peaks in a multiplet are closer than this value, the peaks are
        merged.
    max_perturbation : int, default=1
        Tolerance to add perturbations to the multiplets. See the notes for an
        explanation of its usage.

    Returns
    -------
    annotations : List[List[float]]
        List of valid annotations. Each element is a list of coupling constants
        that give place to the multiplet

    """
    candidates = _find_candidate_heights(height, max_perturbation)
    valid_annotations = list()
    for c_height in candidates:
        j_list = _multiplet_to_j_list(f, c_height, f_tol)
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
    right = height[height.size: q - 1 + r: -1]
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
    valid_deltas = _get_valid_deltas(height, max_delta)
    candidates = list()
    # add perturbations of +/- 1 to height to take into account errors in height
    for delta in valid_deltas:
        for n_pos in range(max_delta + 1):
            for n_neg in range(max_delta + 1 - n_pos):
                n = n_pos + n_neg
                if (n_pos - n_neg) == delta:
                    pert_values = [-1] * n_neg + [1] * n_pos
                    pert_values = np.array(pert_values, dtype=height.dtype)
                    for perm_index in permutations(range(height.size), r=n):
                        perm_index = list(perm_index)
                        pert = height.copy()
                        pert[perm_index] += pert_values

                        # check min values at start/end and multiplet symmetry
                        check_boundaries = (pert[0] == 1) and (pert[-1] == 1)
                        all_pos = (pert > 0).all()

                        if check_boundaries and _is_symmetric(pert) and all_pos:
                            candidates.append(pert)
    return candidates


def _get_valid_deltas(height: np.ndarray, max_delta: int) -> List[int]:
    total_height = height.sum()
    max_exponent = round(log2(total_height)) + 2
    valid_deltas = list()
    for k in range(max_exponent):
        power = 2**k
        delta = power - total_height
        if abs(delta) <= max_delta:
            valid_deltas.append(delta)
    return valid_deltas


def _j_list_to_height(
    j_list: List[float], tol: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a list of coupling constants into a vector of peak frequencies and
    heights.

    Parameters
    ----------
    j_list : list[float]
    tol : float
        Tolerance to merge close frequency values.

    Returns
    -------
    freq : array
        frequencies of each peak in the multiplet
    height : array
        Height og each peak in the multiplet.
    """

    # find all frequency values
    n_j = len(j_list)
    if n_j:
        combinations = cartesian([[-0.5, 0.5] for _ in range(n_j)])
        freq_combinations = np.sort((combinations * j_list).sum(axis=1))
        freq = list()
        height = list()
        f_next = freq_combinations[0]
        h_next = 0
        for f in freq_combinations:
            df = f - f_next
            # merge close peaks
            if df < tol:
                f_next = (f_next * h_next + f) / (h_next + 1)
                h_next += 1
            else:
                freq.append(f_next)
                height.append(h_next)
                f_next = f
                h_next = 1
        freq.append(f_next)
        height.append(h_next)
        freq = np.array(freq)
        height = np.array(height)
    else:
        freq = [0]
        height = [1]
    return freq, height


def _multiplet_to_j_list(
    f: np.ndarray, height: np.ndarray, j_tol: float
) -> Optional[List[float]]:
    n = f.size
    f0 = f[0]
    j_list = list()
    rem_height = height
    for k in range(1, n):
        nj = rem_height[k]

        if nj < 1:
            continue

        # add new j values
        df = f[k] - f0
        j_list.extend([df] * nj)

        # subtract area from j candidates
        fq, hq = _j_list_to_height(j_list, j_tol)
        fq += f0 - fq[0]
        closest = find_closest(f, fq)
        if closest.size <= f.size:
            rem_height = height.copy()
            rem_height[closest] -= hq
            if (rem_height == 0).all():
                # all j found
                break
            elif (rem_height < 0).any():
                # invalid multiplet
                j_list = None
                break
        else:
            j_list = None
            break
    return j_list
