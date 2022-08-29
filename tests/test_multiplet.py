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
    height = np.array([1, 3, 3, 2])
    max_delta = 1
    result = m._find_candidate_heights(height, max_delta)
    expected = np.array([1, 3, 3, 1])
    assert len(result) == 1
    assert np.array_equal(result[0], expected)


def test__find_candidate_heights_asymmetric_ddd():
    height = np.array([1, 1, 1, 1, 2, 1, 1, 1])
    max_delta = 1
    result = m._find_candidate_heights(height, max_delta)
    expected = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    assert len(result) == 1
    assert np.array_equal(result[0], expected)


# all tests for _j_list_to_height should return a valid multiplet
@pytest.mark.parametrize(
    "j_list,tol,expected_f,expected_height",
    [
        ([], 0.5, [0], [1]),  # singlet
        ([4.0], 0.5, [-2.0, 2.0], [1, 1]),      # doublet
        ([4.0, 4.0], 0.5, [-4.0, 0.0, 4.0], [1, 2, 1]),     # triplet
        ([4.0, 4.0, 4.0], 0.5, [-6.0, -2.0, 2.0, 6.0], [1, 3, 3, 1]),  # quartet
        ([4.0, 7.0], 0.5, [-5.5, -1.5, 1.5, 5.5], [1, 1, 1, 1])     # dd
    ]
)
def test__j_list_to_height(j_list, tol, expected_f, expected_height):
    f, height = m._j_list_to_height(j_list, tol)
    expected_f = np.array(expected_f)
    expected_height = np.array(expected_height)
    assert np.array_equal(expected_height, height)
    assert np.allclose(expected_f, f)


def test__multiplet_to_j_list_single_peak():
    f = np.array([0.0])
    height = np.array([1])
    expected = list()
    j_tol = 0.5
    result = m._multiplet_to_j_list(f, height, j_tol)
    assert expected == result


@pytest.mark.parametrize(
    "f,height,j_expected",
    [
        ([0.0, 5.0], [1, 1], [5.0]),    # doublet
        ([0.0, 5.0, 10.0], [1, 2, 1],  [5.0, 5.0]),     # triplet
        ([0.0, 5.0, 10.0, 15.0], [1, 3, 3, 1], [5.0, 5.0, 5.0]),    # quartet
        ([0.0, 5.0, 7.0, 12.0], [1, 1, 1, 1], [5.0, 7.0])   # double doublet
    ]
)
def test__multiplet_to_j_list_valid_multiplet(f, height, j_expected):
    f = np.array(f)
    height = np.array(height)
    j_tol = 0.5
    result = m._multiplet_to_j_list(f, height, j_tol)
    assert np.allclose(j_expected, result)


@pytest.mark.parametrize(
    "f,height",
    [
        ([0.0, 5.0, 10.0], [1, 1, 1]),
        ([0.0, 5.0, 10.0, 15.0], [1, 7, 7, 1])
    ]
)
def test__multiplet_to_j_list_invalid_multiplet(f, height):
    f = np.array(f)
    height = np.array(height)
    j_tol = 0.5
    result = m._multiplet_to_j_list(f, height, j_tol)
    assert result is None
