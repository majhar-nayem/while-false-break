"""Tests for numcompute.optim. Run with:  python -m pytest tests/test_optim.py"""

import numpy as np
import pytest

from numcompute.optim import grad, jacobian
from numcompute.utils import softmax


def rosenbrock(v):
    x, y = v
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rosenbrock_grad(v):
    x, y = v
    return np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])


def test_grad_matches_analytic():
    x = np.array([-1.2, 1.0])
    assert np.allclose(grad(rosenbrock, x), rosenbrock_grad(x), rtol=1e-7)
    assert np.allclose(grad(rosenbrock, x, h=1e-7, method="forward"), rosenbrock_grad(x), rtol=1e-4)
    assert np.allclose(grad(lambda v: np.sum(v**2), [1.0, -2.0, 3.0]), [2, -4, 6])


def test_central_more_accurate_than_forward():
    x = np.array([1.37, 2.11])
    exact = rosenbrock_grad(x)
    err_c = np.abs(grad(rosenbrock, x, h=1e-4) - exact).max()
    err_f = np.abs(grad(rosenbrock, x, h=1e-4, method="forward") - exact).max()
    assert err_c < err_f / 100  # O(h^2) vs O(h)


def test_jacobian():
    def F(v):
        return np.array([v[0] ** 2 * v[1], 5 * v[0] + np.sin(v[1])])

    x = np.array([1.0, 2.0])
    exact = np.array([[2 * x[0] * x[1], x[0] ** 2], [5, np.cos(x[1])]])
    J = jacobian(F, x)
    assert J.shape == (2, 2) and np.allclose(J, exact)
    assert np.allclose(jacobian(F, x, method="forward"), exact, atol=1e-4)
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # linear map: J = A
    assert np.allclose(jacobian(lambda v: A @ v, np.zeros(3)), A)


def test_softmax_jacobian_columns_sum_to_zero():
    J = jacobian(softmax, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(J.sum(axis=0), 0, atol=1e-9)  # outputs always sum to 1


def test_errors():
    with pytest.raises(ValueError):
        grad(rosenbrock, [1.0, 1.0], h=0)
    with pytest.raises(ValueError):
        grad(rosenbrock, [1.0, 1.0], method="backward")
    with pytest.raises(ValueError):  # not a scalar
        grad(lambda v: v, [1.0, 2.0])
    with pytest.raises(ValueError):  # not a vector
        jacobian(lambda v: v.sum(), [1.0, 2.0])
    with pytest.raises(ValueError):
        grad(rosenbrock, [])
    with pytest.raises(ValueError):
        grad(rosenbrock, [[1.0, 2.0]])
    with pytest.raises(TypeError):
        grad(rosenbrock, ["a", "b"])
