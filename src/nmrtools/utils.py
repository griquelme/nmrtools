import numpy as np
from typing import Union


def fid(grid: np.ndarray, t2: float, omega: float, phase: float) -> np.ndarray:
    """
    Computes a free induction decay signal.

    Parameters
    ----------
    grid : array
    t2 : float
        Transverse relaxation
    omega : float
        Larmor frequency
    phase : float
        Phase of the signal, in radians

    Returns
    -------
    np.ndarray

    """
    return np.exp(- grid / t2 + omega * grid * 1j + phase)


def lorentz(grid, omega, t2):
    """
    Computes an absorptive lorentzian

    Parameters
    ----------
    grid : array
    t2 : float
        Transverse relaxation
    omega : float
        Larmor frequency

    Returns
    -------
    np.ndarray

    """
    lam = 1 / t2
    res = lam ** 2 / (lam ** 2 + (grid - omega) ** 2)
    return res


def covmat(X: np.ndarray) -> np.ndarray:
    """
    Covariance matrix of a data matrix

    Parameters
    ----------
    X : array
        Data matrix with shape `(n, p)` where `n` is the number of observations
        and `p` is the number of features.

    Returns
    -------
    cov : np.ndarray
        Covariance matrix with shape `(p, p)`

    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    return np.dot(Xc.T, Xc) / (n - 1)


def covmatk(X: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the k-th row of the covariance matrix of a data matrix.

    Parameters
    ----------
    X : array
        Data matrix with shape `(n, p)` where `n` is the number of observations
        and `p` is the number of features.
    k : int
        Index of a column of X.

    Returns
    -------
    np.ndarray with shape `(p, )`
        k-th row of the covariance matrix

    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    return np.dot(Xc[:, k].reshape((1, n)),  Xc) / (n - 1)


def corrmat(X):
    """
    Correlation matrix of a data matrix

    Parameters
    ----------
    X : array
        Data matrix with shape `(n, p)` where `n` is the number of observations
        and `p` is the number of features.

    Returns
    -------
    corr : np.ndarray

    """
    n = X.shape[0]
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    return np.dot(Z.T, Z) / (n - 1)


def corrmatk(X: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the k-th row of the correlation matrix of a data matrix.

    Parameters
    ----------
    X : array
        Data matrix with shape `(n, p)` where `n` is the number of observations
        and `p` is the number of features.
    k : int
        Index of a column of X.

    Returns
    -------
    np.ndarray with shape `(p, )`
        k-th row of the correlation matrix

    """
    n = X.shape[0]
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    return np.dot(Z[:, k].reshape((1, n)), Z) / (n - 1)


def find_closest(x: np.ndarray, xq: Union[np.ndarray, float, int],
                 is_sorted: bool = True) -> np.ndarray:
    """
    Search the closest value between two arrays.
    Parameters
    ----------
    x : array
        Sorted array to perform search
    xq : array
        query values
    is_sorted : bool, default=True
        If True, assumes that x is sorted.
    Returns
    -------
    array of indices in x
    """
    if is_sorted:
        return _find_closest_sorted(x, xq)
    else:
        sorted_index = np.argsort(x)
        closest_index = _find_closest_sorted(x[sorted_index], xq)
        return sorted_index[closest_index]


def _find_closest_sorted(x: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """
    Find the index in x closest to each xq element. Assumes that x is sorted.
    Parameters
    ----------
    x: numpy.ndarray
        Sorted vector
    xq: numpy.ndarray
        search vector
    Returns
    -------
    ind: numpy.ndarray
        array with the same size as xq with indices closest to x.
    Raises
    ------
    ValueError: when x or xq are empty.
    """

    if x.size == 0:
        msg = "`x` must be a non-empty array."
        raise ValueError(msg)

    if not isinstance(xq, np.ndarray):
        xq = np.array([xq])

    ind = np.searchsorted(x, xq)

    # cases where the index is between 1 and x.size - 1
    mask = (ind > 0) & (ind < x.size)
    ind[mask] -= (xq[mask] - x[ind[mask] - 1]) < (x[ind[mask]] - xq[mask])
    # when the index is x.size, then the closest index is x.size -1
    ind[ind == x.size] = x.size - 1
    return ind
