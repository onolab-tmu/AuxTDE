# -*- coding: utf-8 -*-
import numpy as np


def GCC(xspec):
    """
    Compute a time delay estimate based on discrete GCC.

    parameters
    ----------
    xspec: array_like (n_freq, )
        Cross-spectrum

    return
    ----------
    discrete sample time delay estimate
    """

    # Compute cross correlation
    corr = np.real(np.fft.irfft(xspec))

    # Get and return the location of its maximum
    return np.argmax(corr)


def GCC_with_parafit(xspec, ret_GCC=False):
    """
    Compute the sub-sample estimate by the quadratic interpolation of GCC.

    parameters
    ----------
    xspec: array_like (n_freq, )
        Cross-spectrum
    ret_GCC: bool, optional
        If True, return the discrete estimate too

    return
    ----------
    sub-sample time delay estimate
    if ret_GCC: discrete estimate too
    """

    # Compute cross correlation
    corr = np.real(np.fft.irfft(xspec))

    # Get location of its maximum
    t_max = np.argmax(corr)

    # Get the value of three points around the maximum
    if t_max == corr.shape[0] - 1:
        y_vec = [corr[t_max - 1], corr[t_max], corr[0]]
    else:
        y_vec = [corr[t_max - 1], corr[t_max], corr[t_max + 1]]

    # return
    if ret_GCC:
        return parafit(t_max, y_vec), t_max
    else:
        return parafit(t_max, y_vec)


def parafit(x_2, y_vec):
    """
    parabolic interpolation: compute x coordinate of the vertex
    of the quadratic function determined by the given three coordinates.

    parameters
    ----------
    x_2: float
        Discrete x coordinate of the extremum, i.e., x_2 in the three points [x_1, x_2, x_3]
        x_2 must be the median of the three adjacent discrete points
    y_vec: array_like (3, )
        y coordinates for the interpolation, y_vec = [y_1, y_2, y_3]'

    return
    ----------
    x coordinate of the vertex
    """

    x_vtx = -0.5 * (y_vec[2] - y_vec[0]) / (y_vec[2] - 2 * y_vec[1] + y_vec[0])
    return x_vtx + x_2


def para_coef(x, y):
    """
    Compute coefficients of a quadratic function

    parameters
    ----------
    x: array_like (3, )
        x coordinates for the interpolation, x = [x_1, x_2, x_3]'
    y: array_like (3, )
        y coordinates for the interpolation, y = [y_1, y_2, y_3]'

    return
    ----------
    a, b, c: float
        coefficients of the quadratic function
    """

    x32 = x[2] - x[1]
    x32p = x[2] + x[1]
    x31 = x[2] - x[0]
    x12 = x[0] - x[1]
    y32 = y[2] - y[1]
    y31 = y[2] - y[0]

    a = (x32 * y31 - x31 * y32) / x31 * x32 * x12
    b = y32 / x32 - a * x32p
    c = y[0] - a * x[0] ^ 2 - b * x[0]

    return a, b, c


def para_func(x, a, b, c):
    y = a * x ** 2 + b * x + c
    return y
