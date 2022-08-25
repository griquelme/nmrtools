import numpy as np
import pytest
from nmrtools import utils


def test_find_closest_empty_x():
    x = np.array([])
    y = 10
    with pytest.raises(ValueError):
        utils.find_closest(x, y)


def test_find_closest_left_border():
    x = np.arange(10)
    y = -1
    result = utils.find_closest(x, y)
    expected = np.array([0])
    assert np.array_equal(result, expected)


def test_find_closest_right_border():
    x = np.arange(10)
    y = 10
    result = utils.find_closest(x, y)
    expected = np.array([(x.size - 1)])
    assert np.array_equal(result, expected)


def test_find_closest_middle():
    x = np.arange(10)
    y = 4.6
    result = utils.find_closest(x, y)
    expected = np.array([5])
    assert np.array_equal(result, expected)


def test_find_closest_empty_y():
    x = np.arange(10)
    y = np.array([])
    result = utils.find_closest(x, y)
    expected = np.array([])
    assert np.array_equal(result, expected)


def test_find_closest_multiple_values():
    x = np.arange(100)
    y = np.array([-10, 4.6, 67.1, 101])
    expected = np.array([0, 5, 67, 99], dtype=int)
    result = utils.find_closest(x, y)
    assert np.array_equal(result, expected)
