import numpy as np
from . import multiplet
from typing import Dict, List, Tuple


def simulate_compound_time(
    t: np.ndarray,
    f0: List[float],
    j: List[List[float]],
    t2: List[float]
) -> np.ndarray:
    """
    Simulates a compound as a linear combination of multiplets.

    Parameters
    ----------
    t : array
        Time samples
    f0 : List[float]
        Larmor frequency of each multiplet, in `1 / t` units.
    j : List[List[float]]
        Coupling constants for each multiplet, in `1 / t` units. `j` must have
        the same length as `f0`. For singlets, an empty list must be used.
    t2 : List[float]
        Transverse relaxation time for each multiplet, in `t` units.

    Returns
    -------
    fid : array
        Time domain signal of the compound

    Examples
    --------
    simulate a compound with two multiplets: a singlet and a doublet:
    >>> import numpy as np
    >>> from nmrtools.simulation import simulate_compound_time
    >>> t = np.linspace(0, 4, 500)
    >>> f0 = [20, 40]
    >>> j = [[], [5]]
    >>> t2 = [1.0, 1.5]
    >>> fid = simulate_compound_time(t, f0, j, t2)

    """
    fid = np.zeros(t.size, dtype="complex128")
    for f0k, jk, t2k in zip(f0, j, t2):
        fid += multiplet.simulate_time(t, jk, t2k, f0=f0k)
    return fid


def simulate_compound_frequency(
    f: np.ndarray,
    f0: List[float],
    j: List[List[float]],
    t2: List[float]
) -> np.ndarray:
    """
    Simulates a compound as a linear combination of multiplets.

    Parameters
    ----------
    f : array
        frequency samples
    f0 : List[float]
        Larmor frequency of each multiplet, in `f` units.
    j : List[List[float]]
        Coupling constants for each multiplet, in `f` units. `j` must have
        the same length as `f0`. For singlets, an empty list must be used.
    t2 : List[float]
        Transverse relaxation time for each multiplet, in `1 / f` units.

    Returns
    -------
    sp : array
        Frequency domain signal of the compound

    Examples
    --------
    simulate a compound with two multiplets: a singlet and a doublet:
    >>> import numpy as np
    >>> from nmrtools.simulation import simulate_compound_frequency
    >>> t = np.linspace(0, 4, 500)
    >>> f0 = [20, 40]
    >>> j = [[], [5]]
    >>> t2 = [1.0, 1.5]
    >>> sp = simulate_compound_frequency(t, f0, j, t2)

    """
    sp = np.zeros(f.size, dtype="float64")
    for f0k, jk, t2k in zip(f0, j, t2):
        sp += multiplet.simulate_frequency(f, jk, t2k, f0=f0k)
    return sp
