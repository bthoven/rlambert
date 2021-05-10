# -*- coding: utf-8 -*-
"""
Computes the r-Lambert number `W_r` for given `x` and `r`.

It's based on the C++ code ´rLambert.cpp´ created by István Mezö and hosted at:

    https://github.com/IstvanMezo/r-Lambert-function

"""


import math as mt
import sys

try:
    from numba import njit, prange
    _NUMBA_INSTALLED = True
except ModuleNotFoundError:
    _NUMBA_INSTALLED = False
import numpy as np


__author__ = "Beethoven Santos (thovensantos@gmail.com)"
__version__ = "21.05.10"

# Constants
_EPS = sys.float_info.epsilon
_EULER_NUMBER = mt.e
_INVERSE_OF_e = 1 / _EULER_NUMBER
_INVERSE_OF_e2 = mt.exp(-2)
_ABS_TOL_FOR_ZERO = 1e-9


class _ConvergenceError(Exception):
    """Error exception to be raised when a convergence wasn't met."""
    pass


# This sets `njit` to a dummy decorator and `prange` to the Python built-in
# `range` function to avoid errors by using `njit` and `prange` when numba
# isn't installed.
if not _NUMBA_INSTALLED:
    def phantom_decorator(fastmath=False):
        def inner(func):
            return func
        return inner
    njit = phantom_decorator
    prange = range


def _is_zero(x):
    """A wrapper to the math.isclose function for comparing `x` to zero."""
    return mt.isclose(x, 0, abs_tol=_ABS_TOL_FOR_ZERO)


@njit(fastmath=True)
def _is_zero_jit(x):
    """Function for comparing `x` to zero."""
    return abs(x) <= _ABS_TOL_FOR_ZERO


def _get_lhs(x, r):
    """Left-hand side of the equation x * exp(x) + x * r = z."""
    f = x * (mt.exp(x) + r)
    dfdx = (1 + x) * mt.exp(x) + r
    d2fdx2 = (2 + x) * mt.exp(x)
    return f, dfdx, d2fdx2


@njit(fastmath=True)
def _get_lhs_jit(x, r):
    """Left-hand side of the equation x * exp(x) + x * r = z."""
    f = x * (mt.exp(x) + r)
    dfdx = (1 + x) * mt.exp(x) + r
    d2fdx2 = (2 + x) * mt.exp(x)
    return f, dfdx, d2fdx2


def _decrement(variable, x, r):
    """Decrement applied on W_r at each iteration in `_compute_w` function."""
    f, df, df2 = _get_lhs(variable, r)
    return 2 * ((f - x) * df) / (2 * (df ** 2) - (f - x) * df2)


@njit(fastmath=True)
def _decrement_jit(variable, x, r):
    """Decrement applied on W_r at each iteration in `_compute_w` function."""
    f, df, df2 = _get_lhs_jit(variable, r)
    return 2 * ((f - x) * df) / (2 * (df ** 2) - (f - x) * df2)


def _compute_w(x, r, w_ini, w_prev, precision, maxiter):
    """
    Compute `W_r` for given `x` and `r`.

    Parameters
    ----------
    x : float
        Independent variable at which `W_r` will be computed.
    r : float
        r-Parameter of the equation which defines the r-Lambert function `W_r`.
    w_ini : float
        Initial value of the r-Lambert `W_r` function.
    w_prev : float
        Previous value of the r-Lambert `W_r` function.
    precision : float
        Tolerance that will be considered to check if the value of `W_r` has
        converged.
    maxiter : int
        Maximum number of iterations to compute `W_r`.

    Returns
    -------
    w, w_prev : tuple (float)
        The current and previous values of the r-Lambert function `W_r`.

    """
    w = w_ini
    iterator = 0
    while (abs(w - w_prev) > precision) and (iterator < maxiter):
        w_prev = w
        w -= _decrement(w, x, r)
        iterator += 1
    return w, w_prev


@njit(fastmath=True)
def _compute_w_jit(x, r, w_ini, w_prev, precision, maxiter):
    """
    Compute `W_r` for given `x` and `r`.

    Parameters
    ----------
    x : float
        Independent variable at which `W_r` will be computed.
    r : float
        r-Parameter of the equation which defines the r-Lambert function `W_r`.
    w_ini : float
        Initial value of the r-Lambert `W_r` function.
    w_prev : float
        Previous value of the r-Lambert `W_r` function.
    precision : float
        Tolerance that will be considered to check if the value of `W_r` has
        converged.
    maxiter : int
        Maximum number of iterations to compute `W_r`.

    Returns
    -------
    w, w_prev : tuple (float)
        The current and previous values of the r-Lambert function `W_r`.

    """
    w = w_ini
    iterator = 0
    while (abs(w - w_prev) > precision) and (iterator < maxiter):
        w_prev = w
        w -= _decrement_jit(w, x, r)
        iterator += 1
    return w, w_prev


def _halley_iteration_for_W(z, max_iterations=10):
    converged = False

    # Initial approx for iteration...
    if z < 1:  # Series near 0
        p = mt.sqrt(2 * (_EULER_NUMBER * z + 1))
        w = p * (p * (-(1 / 3) + p * 11 / 72) + 1) - 1
    else:
        w = mt.log(z / mt.log(z))  # Asymptotic

    for _ in range(max_iterations):  # Halley iteration
        e = mt.exp(w)
        t = w * e - z
        p = w + 1
        t /= e * p - 0.5 * (p + 1) * t / p
        w -= t
        if abs(t) < _EPS * (1 + abs(w)):  # Rel-abs error
            converged = True
            break

    if not converged:
        raise _ConvergenceError

    return w


@njit(fastmath=True)
def _halley_iteration_for_W_jit(z, max_iterations=10):
    converged = False

    # Initial approx for iteration...
    if z < 1:  # Series near 0
        p = mt.sqrt(2 * (_EULER_NUMBER * z + 1))
        w = p * (p * (-(1 / 3) + p * 11 / 72) + 1) - 1
    else:
        w = mt.log(z / mt.log(z))  # Asymptotic

    for _ in prange(max_iterations):  # Halley iteration
        e = mt.exp(w)
        t = w * e - z
        p = w + 1
        t /= e * p - 0.5 * (p + 1) * t / p
        w -= t
        if abs(t) < _EPS * (1 + abs(w)):  # Rel-abs error
            converged = True
            break

    if not converged:
        raise _ConvergenceError

    return w


def _halley_iteration_for_Wminus1(z, max_iterations=10):
    p = 1
    converged = False

    # Initial approx for iteration...
    if z < -1e-6:  # Series about -1/e
        p = -mt.sqrt(2 * (_EULER_NUMBER * z + 1))
        w = p * (p * (-(1 / 3) + p * 11 / 72) + 1) - 1
    else:  # Asymptotic near zero
        l1 = mt.log(-z)
        l2 = mt.log(-l1)
        w = l1 - l2 + l2 / l1

    if abs(p) >= 1e-4:
        for _ in range(max_iterations):
            e = mt.exp(w)
            t = w * e - z
            p = w + 1
            t /= e * p - 0.5 * (p + 1) * t / p
            w -= t
            if abs(t) < _EPS * (1 + abs(w)):  # Rel-abs error
                converged = True
                break

    if not converged:
        raise _ConvergenceError

    return w


@njit(fastmath=True)
def _halley_iteration_for_Wminus1_jit(z, max_iterations=10):
    p = 1
    converged = False

    # Initial approx for iteration...
    if z < -1e-6:  # Series about -1/e
        p = -mt.sqrt(2 * (_EULER_NUMBER * z + 1))
        w = p * (p * (-(1 / 3) + p * 11 / 72) + 1) - 1
    else:  # Asymptotic near zero
        l1 = mt.log(-z)
        l2 = mt.log(-l1)
        w = l1 - l2 + l2 / l1

    if abs(p) >= 1e-4:
        for _ in prange(max_iterations):
            e = mt.exp(w)
            t = w * e - z
            p = w + 1
            t /= e * p - 0.5 * (p + 1) * t / p
            w -= t
            if abs(t) < _EPS * (1 + abs(w)):  # Rel-abs error
                converged = True
                break

    if not converged:
        raise _ConvergenceError

    return w


def _lambertW(z, max_halley_iterations=10, do_jit=True):
    if z < -_INVERSE_OF_e:
        print(f"[func _lambertW]: Bad argument z={z}. "
              "`z` cannot be lower than -e⁻¹ for the main branch W₀. Exiting.")
        return None

    if do_jit:
        w_func = _halley_iteration_for_W_jit
        is_zero_func = _is_zero_jit
    else:
        w_func = _halley_iteration_for_W
        is_zero_func = _is_zero

    if is_zero_func(z):
        return 0

    if z < -_INVERSE_OF_e + 1e-4:  # Series near -1/e in sqrt(q)
        q = z + _INVERSE_OF_e
        r = mt.sqrt(q)
        q2 = q * q
        q3 = q2 * q
        A = 2.331643981597124203363536062168
        B = - 1.812187885639363490240191647568
        C = 1.936631114492359755363277457668
        D = - 2.353551201881614516821543561516
        E = 3.066858901050631912893148922704
        F = - 4.175335600258177138854984177460
        G = 5.858023729874774148815053846119
        H = - 8.401032217523977370984161688514
        return (- 1 + A * r + B * q + C * r * q + D * q2 + E * r * q2 + F * q3
                + G * r * q3 + H * q3 * q)  # error approx: 1e-16

    try:
        w = w_func(z, max_iterations=max_halley_iterations)
    except _ConvergenceError:
        print(f"[func _lambertW]: No convergence at z={z}, exiting.")
        return None

    return w


def _lambertWminus1(z, max_halley_iterations=10, do_jit=True):
    if z < -_INVERSE_OF_e or z >= 0:
        print(f"[func _lambertWminus1]: Bad argument z={z}. "
              "`z` must be inside the interval [-e⁻¹, 0) for W₋₁. Exiting.")
        return None

    try:
        w_func = (
            _halley_iteration_for_Wminus1_jit if do_jit
            else _halley_iteration_for_Wminus1
        )
        w = w_func(z, max_iterations=max_halley_iterations)
    except _ConvergenceError:
        print(f"[func _lambertWminus1]: No convergence at z={z}, exiting.")
        return None

    return w


def rlambert(x, r, precision=9, max_iterations=20, max_halley_iterations=10,
             use_numba=True):
    do_jit = use_numba
    if use_numba and not _NUMBA_INSTALLED:
        print("Tried to use numba, but it isn't installed. "
              "Using pure Python instead.")
        do_jit = False

    if do_jit:
        compute_w_func = _compute_w_jit
        lhs_func = _get_lhs_jit
        is_zero_func = _is_zero_jit
    else:
        compute_w_func = _compute_w
        lhs_func = _get_lhs
        is_zero_func = _is_zero

    if is_zero_func(x):  # W(0, r) = 0 always
        return 0

    if is_zero_func(r):
        if (x < 0):
            print(f"There is one additional branch for x={x} and r={r}: ")
            print(f"  W₋₁({x}, 0) = "
                  f"{_lambertWminus1(x, max_halley_iterations, do_jit)}")
        return _lambertW(x, max_halley_iterations, do_jit)

    # If r >= e⁻², there is just one branch of W(x, r):
    if r >= _INVERSE_OF_e2:
        if (r == _INVERSE_OF_e2) and (x == -4 * _INVERSE_OF_e2):
            return -2
        # At this x, W(x, e⁻²) is not differentiable, so Halley's method does
        # not work. But it can be calculated that W(x, e⁻²) = -2. If x is close
        # but not equal to -4e⁻², there is no problem. For example, if
        # x = -4e⁻² ± 10⁻¹¹, the program still gives the correct result.

        # Begin the iteration up to `precision`
        w_prev = mt.inf
        # Initial value for the Halley method
        if x > 1:
            w_ini = mt.log(x) - mt.log(mt.log(x))
        if x < -1:
            w_ini = 1 / r * x
        if abs(x) <= 1:
            w_ini = 0

        w = compute_w_func(x, r, w_ini, w_prev, precision, max_iterations)[0]

        return w

    # Here comes the calculation when W(x, r) has three branches
    # (i.e. 0 < r < e⁻²).
    elif 0 < r < _INVERSE_OF_e2:
        # Left branch point
        alpha = _lambertWminus1(-r * _EULER_NUMBER, max_halley_iterations,
                                do_jit) - 1
        beta = _lambertW(-r * _EULER_NUMBER, max_halley_iterations,
                         do_jit) - 1  # Right branch point
        falpha = lhs_func(alpha, r)[0]
        fbeta = lhs_func(beta, r)[0]

        if x < fbeta:  # Leftmost branch
            if x < -40.54374295204823:
                return x / r  # Because x * e^x < 10⁻¹⁶ as x < -40.5437...
            w_prev = mt.inf
            w_ini = x / r
            w = compute_w_func(x, r, w_ini, w_prev, precision,
                               max_iterations)[0]
            return w

        if x >= fbeta and x <= falpha:  # Leftmost and inner branches
            # Halley iteration for the leftmost branch
            wm2_ini = x / r  # Initial value for the leftmost branch W_2(x, r)
            wm2, w_prev = compute_w_func(x, r, wm2_ini, w_prev, precision,
                                         max_iterations)
            print(f"There are two additional branches for x={x} and r={r}:")
            print(f"  W₋₂({x}, {r}) = {wm2}")

            # Halley iteration for the inner branch
            wm1_ini = -3  # Initial value for the inner branch W₋₁(x, r)
            wm1, w_prev = compute_w_func(x, r, wm1_ini, w_prev, precision,
                                         max_iterations)
            print(f"  W₋₁({x}, {r}) = {wm1}")

            # Halley iteration for the rightmost branch
            wm0_ini = -1  # Initial value for the rightmost branch W₀(x, r)
            wm0 = compute_w_func(x, r, wm0_ini, w_prev, precision,
                                 max_iterations)[0]

            return wm0

        if x > falpha:  # Rightmost branch
            # Initial value
            w_prev = mt.inf
            w_ini = mt.log(x) - mt.log(mt.log(x)) if x > 1 else 0
            w = compute_w_func(x, r, w_ini, w_prev, precision,
                               max_iterations)[0]
            return w

    elif r < 0:  # Two branches separated by W(-r*e) - 1
        # Minimum of f: W(-r*e) - 1, zeros: 0 and log(-r)
        if x < lhs_func(
            _lambertW(-r * _EULER_NUMBER, max_halley_iterations, do_jit) - 1, r
        )[0]:
            print(f"First argument x={x} of _lambertW(x, r) is out of domain")
            return None
        if x == mt.log(-r):
            return 0

        # Two initial values, one less than the minimum of f, the other one is
        # greater.
        if x < 0:
            # Left branch
            w_prev = mt.inf
            w_ini = _lambertW(-r * _EULER_NUMBER, max_halley_iterations,
                              do_jit) - 2
            w = compute_w_func(x, r, w_ini, w_prev, precision,
                               max_iterations)[0]
            print(f"There is one additional branch for x={x} and r={r}:")
            print(f"  W₋₁({x}, {r}) = {w}")

            # Right branch
            w_prev = mt.inf
            w_ini = _lambertW(-r * _EULER_NUMBER, max_halley_iterations,
                              do_jit)
            w = compute_w_func(x, r, w_ini, w_prev, precision,
                               max_iterations)[0]
            return w

        if x > 0:
            lzero, rzero = (mt.log(-r), 0) if r > -1 else (0, mt.log(-r))

            # Left branch
            w_prev = mt.inf
            w_ini = lzero - 1
            w = compute_w_func(x, r, w_ini, w_prev, precision,
                               max_iterations)[0]
            print(f"There is one additional branch for x={x} and r={r}:")
            print(f"  W₋₁({x}, {r}) = {w}")

            # Right branch
            w_prev = mt.inf
            w_ini = rzero + 1
            w = compute_w_func(x, r, w_ini, w_prev, precision,
                               max_iterations)[0]
            return w

    # Should never get here
    print(f"[func rlambert]: No convergence at x={x}, exiting.")
    return None


if __name__ == "__main__":
    precision = 1e-9
    xr = [
        [1.35, 2.7],
        [0, 1],
        [-0.3, 0],
        [2, 0],
        [-4 * mt.exp(2), mt.exp(2)],
        [1, 0.5]
    ]
    for x, r in xr:
        print(f"W({x:g}, {r:g}) = {rlambert(x, r, precision)}")
        print()
