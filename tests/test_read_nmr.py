# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:34:42 2022

@author: CIBION-NMR
"""

# import  read_nmr
# import numpy as np
# import nmrglue as ng
#
# def test_guess_zero_fill_f1():
#     TD1 = 40
#     TD2 = 8192
#     SW1 = 78.125
#     SW2 = 10000
#     DS = 16
#     n_f2_pad = 4096
#     expected_n_pad_f1 = 24
#     test_n_pad_f1 = read_nmr.guess_zero_fill_f1(SW1, SW2, TD1, TD2, n_f2_pad)
#     assert expected_n_pad_f1 == test_n_pad_f1
#
#
# def test_pad_fid():
#     path = "243"
#     dic, X = ng.fileio.bruker.read(path)
#     row, col = X.shape
#     n_f1 = 1000
#     n_f2 = 5000
#     X_pad = read_nmr.pad_fid(X, n_f1, n_f2)
#     expected_row = n_f1
#     expected_col = n_f2
#     test_row, test_col = X_pad.shape
#     assert expected_row == test_row
#     assert expected_col == test_col
#     assert np.array_equal(X, X_pad[:row, :col])