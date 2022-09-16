import numpy as np
from . import multiplet
from typing import List


def simulate_compound_time(
    t: np.ndarray,
    f0: List[float],
    j: List[List[float]],
    t2: List[float],
    abundance: List[int]
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
    abundance : List[int]
        Number of nucleii that generates each multiplet.

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
    >>> abundance = [1, 1]
    >>> fid = simulate_compound_time(t, f0, j, t2, abundance)

    """
    fid = np.zeros(t.size, dtype="complex128")
    for f0k, jk, t2k, nk in zip(f0, j, t2, abundance):
        fid += nk * multiplet.simulate_time(t, jk, t2k, f0=f0k)
    return fid


def simulate_compound_frequency(
    f: np.ndarray,
    f0: List[float],
    j: List[List[float]],
    t2: List[float],
    abundance: List[int]
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
    abundance : List[int]
        Number of nucleii that generates each multiplet.

    Returns
    -------
    sp : array
        Frequency domain signal of the compound

    Examples
    --------
    simulate a compound with two multiplets: a singlet and a doublet:
    >>> import numpy as np
    >>> from nmrtools.simulation import simulate_compound_frequency
    >>> f = np.linspace(0, 4, 500)
    >>> f0 = [20, 40]
    >>> j = [[], [5]]
    >>> t2 = [1.0, 1.5]
    >>> abundance = [1, 1]
    >>> sp = simulate_compound_frequency(f, f0, j, t2, abundance)

    """
    sp = np.zeros(f.size, dtype="float64")
    for f0k, jk, t2k, nk in zip(f0, j, t2, abundance):
        mk = multiplet.simulate_frequency(f, jk, t2k, f0=f0k)
        sp += nk * mk / max(mk)
    return sp


def make_mixture_time(
    t: np.ndarray,
    f0: List[List[float]],
    j: List[List[List[float]]],
    t2: List[List[float]],
    abundance: List[List[int]]
) -> np.ndarray:
    """
    Creates an array where each row is compound and each column is a frequency.

    Parameters
    ----------
    t : array
        time samples
    f0 : List[List[float]]
        A list where each item contains the Larmor frequencies for each
        compound. See the arguments for `simulate_compound_frequency`.
    j : List[List[float]]
        A list where each item contains the coupling constants for the
        multiplets in each compound. See the arguments for
        `simulate_compound_frequency`.
    t2 : List[float]
        A list where each item contains the transverse relaxation time for the
        multiplets of each compound.
    abundance : List[List[int]]
        Number of nucleii that generates each multiplet.


    Returns
    -------
    mixture: array
        An array where each column is a simulated spectrum in the mixture and
        each column is a frequency value.

    Examples
    --------
    Creates a mixture of two compounds: one with two singlets and one with a
    doublet.
    >>> import numpy as np
    >>> from nmrtools.simulation import make_mixture_time
    >>> t = np.linspace(0, 4, 500)
    >>> f0 = [[20, 40], [30]]
    >>> j = [[[], []], [[5]]]
    >>> t2 = [[1.0, 1.5], [1.5]]
    >>> abundance = [[1, 1], [1]]
    >>> sp = make_mixture_time(t, f0, j, t2, abundance)

    """
    n_compounds = len(f0)
    mixture = np.zeros(shape=(n_compounds, t.size), dtype="complex128")
    for k in range(n_compounds):
        mixture[k] = simulate_compound_time(t, f0[k], j[k], t2[k], abundance[k])
    return mixture


def make_mixture_frequency(
    f: np.ndarray,
    f0: List[List[float]],
    j: List[List[List[float]]],
    t2: List[List[float]],
    abundance: List[List[int]]
) -> np.ndarray:
    """
    Creates an array where each row is compound and each column is a frequency.

    Parameters
    ----------
    f : array
        frequency samples
    f0 : List[List[float]]
        A list where each item contains the Larmor frequencies for each
        compound. See the arguments for `simulate_compound_frequency`.
    j : List[List[float]]
        A list where each item contains the coupling constants for the
        multiplets in each compound. See the arguments for
        `simulate_compound_frequency`.
    t2 : List[float]
        A list where each item contains the transverse relaxation time for the
        multiplets of each compound.
    abundance : List[List[int]]
        Number of nucleii that generates each multiplet.


    Returns
    -------
    mixture: array
        An array where each column is a simulated spectrum in the mixture and
        each column is a frequency value.

    Examples
    --------
    Creates a mixture of two compounds: one with two singlets and one with a
    doublet.
    >>> import numpy as np
    >>> from nmrtools.simulation import make_mixture_frequency
    >>> f = np.linspace(0, 4, 500)
    >>> f0 = [[20, 40], [30]]
    >>> j = [[[], []], [[5]]]
    >>> t2 = [[1.0, 1.5], [1.5]]
    >>> abundance = [[1, 1], [1]]
    >>> sp = make_mixture_frequency(f, f0, j, t2, abundance)

    """
    n_compounds = len(f0)
    mixture = np.zeros(shape=(n_compounds, f.size), dtype=float)
    for k in range(n_compounds):
        mixture[k] = simulate_compound_frequency(
            f, f0[k], j[k], t2[k], abundance[k]
        )
    return mixture
