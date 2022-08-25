# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:36:06 2022

@author: CIBION-NMR
"""

import numpy as np
# import nmrglue as ng


def guess_zero_fill_f1(
    SW1: int,
    SW2: int,
    TD1: int,
    TD2: int,
    n_f2_pad: int
) -> int:
    df2 = (SW2 / n_f2_pad)
    n_f1_pad = round(SW1 / df2 - TD1)
    return n_f1_pad


def pad_fid(X: np.ndarray, n_f1: int, n_f2: int) -> np.ndarray:
    """
    Pad data with zeros until specified size.
    
    
    Parameters
    ----------
    X : array
        FID data
    n_f1 : int
        Final number of rows in X.
    n_f2 : int
        Final number of columns in X.
        
    Returns
    -------
    X_pad : array
    
    """
    rows, cols = X.shape
    new_shape = (n_f1, n_f2)
    X_pad = np.zeros(new_shape, dtype=X.dtype)
    X_pad[:rows, :cols] = X
    return X_pad


def exponential_window(n: int, lb: float) -> np.ndarray:
    """

    Parameters
    ----------
    n
    lb

    Returns
    -------

    """
    grid = np.arange(n)
    window = np.exp(- grid * lb) 
    return window


def gaussian_window(n: int, sigma: float):
    grid = np.arange(n)
    window = np.exp(- (grid / sigma) ** 2) 
    return window


def sine_window(n: int, phase: float) -> np.ndarray:
    grid = np.arange(n)
    window = np.sin((np.pi - phase) * grid / n + phase)    
    return window


def apodize(X: np.ndarray, mode: str, axis: int, **params) -> np.ndarray:
    
    row, col = X.shape
    if axis == 0:
        n = col
    elif axis == 1:
        n = row
    
    if mode == "sine":
        window = sine_window(n, **params)
    elif mode == "exponential":
        window = exponential_window(n, **params)
    elif mode == "gaussian":
        window = gaussian_window(n, **params)
    else:
        msg = "valid windows are `sine` or `exponential`."
        raise ValueError(msg)
        
    if axis == 1:
        window = np.reshape(window, (window.size, 1))
        
    X_apod = X * window
    return X_apod


def fourier_transform(X: np.ndarray, axis: int) -> np.ndarray:
    sp = np.fft.fft(X, axis=axis)
    return sp
    