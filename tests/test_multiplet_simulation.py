import numpy as np
import pytest
from nmrtools import multiplet as m


def test__is_symmetric_multiplet_even_length():
    height = np.array([1, 1])
    assert m._is_symmetric(height)


def test__is_symmetric_multiplet_odd_length():
    height = np.array([1, 1, 1])
    assert m._is_symmetric(height)


def test__is_symmetric_multiplet_quartet():
    height = np.array([1, 4, 4, 1])
    assert m._is_symmetric(height)


def test__is_symmetric_multiplet_quintuplet():
    height = np.array([1, 4, 6, 4, 1])
    assert m._is_symmetric(height)


def test__is_symmetric_multiplet_double_doublet():
    height = np.array([1, 1, 1, 1])
    assert m._is_symmetric(height)


def test__is_symmetric_multiplet_asymmetric_even_length():
    height = np.array([1, 1, 2])
    assert not m._is_symmetric(height)


def test__is_symmetric_multiplet_asymmetric_odd_length():
    height = np.array([1, 1, 2, 1])
    assert not m._is_symmetric(height)


def test__find_candidate_heights_symmetric_doublet():
    height = np.array([1, 1])
    max_delta = 1
    result = m._find_candidate_heights(height, max_delta)
    expected = np.array([1, 1])
    assert len(result) == 1
    assert np.array_equal(result[0], expected)


def test__find_candidate_heights_asymmetric_doublet():
    height = np.array([2, 1])
    max_delta = 1
    result = m._find_candidate_heights(height, max_delta)
    expected = np.array([1, 1])
    assert len(result) == 1
    assert np.array_equal(result[0], expected)


def test__find_candidate_heights_asymmetry_greater_than_max_delta():
    height = np.array([3, 1])
    max_delta = 1
    result = m._find_candidate_heights(height, max_delta)
    assert len(result) == 0


def test__find_candidate_heights_symmetric_triplet():
    height = np.array([1, 2, 1])
    max_delta = 0
    result = m._find_candidate_heights(height, max_delta)
    expected = np.array([1, 2, 1])
    assert len(result) == 1
    assert np.array_equal(result[0], expected)


def test__find_candidate_heights_asymmetric_quartet():
    height = np.array([1, 2, 2, 2])
    max_delta = 1
    result = m._find_candidate_heights(height, max_delta)
    expected = np.array([1, 2, 2, 1])
    assert len(result) == 1
    assert np.array_equal(result[0], expected)


def test__find_candidate_heights_asymmetric_ddd():
    height = np.array([1, 1, 1, 1, 2, 1])
    max_delta = 1
    result = m._find_candidate_heights(height, max_delta)
    expected = np.array([1, 1, 1, 1, 1, 1])
    assert len(result) == 1
    assert np.array_equal(result[0], expected)


# all tests for _j_list_to_height should return a valid multiplet

def test__j_list_to_height_doublet():
    j_list = [4.0]
    j_tol = 0.5

    f, height = m._j_list_to_height(j_list, j_tol)
    expected_f = np.array([-2.0, 2.0])
    expected_height = np.array([1, 1])
    assert np.array_equal(expected_height, height)
    assert np.allclose(expected_f, f)


def test__j_list_to_height_triplet():
    j_list = [4.0, 4.0]
    j_tol = 0.5

    f, height = m._j_list_to_height(j_list, j_tol)
    expected_f = np.array([-4.0, 0.0, 4.0])
    expected_height = np.array([1, 2, 1])
    assert np.array_equal(expected_height, height)
    assert np.allclose(expected_f, f)


def test__j_list_to_height_quartet():
    j_list = [4.0, 4.0, 4.0]
    j_tol = 0.5

    f, height = m._j_list_to_height(j_list, j_tol)
    expected_f = np.array([-6.0, -2.0, 2.0, 6.0])
    expected_height = np.array([1, 3, 3, 1])
    assert np.array_equal(expected_height, height)
    assert np.allclose(expected_f, f)


def test__j_list_to_height_double_doublet():
    j_list = [4.0, 7.0]
    j_tol = 0.5

    f, height = m._j_list_to_height(j_list, j_tol)
    expected_f = np.array([-5.5, -1.5, 1.5, 5.5])
    expected_height = np.array([1, 1, 1, 1])
    assert np.array_equal(expected_height, height)
    assert np.allclose(expected_f, f)
