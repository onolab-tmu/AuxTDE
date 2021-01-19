# -*- coding: utf-8 -*-
import numpy as np
import tdoa
from functions.STFT import mSTFT


def aux_tdoa(x, frlen=None, frsft=None, n_iter=10, t0=None, phat=False, ret_all=False):
    """
    Auxilliary function based sub-sample time delay estimation for two signals.
    Ref: Kouei Yamaoka, Robin Scheibler, Nobutaka Ono, and Yukoh Wakabayashi,
         "Sub-Sample Time Delay Estimation via Auxiliary-Function-Based
         Iterative Updates," Proc. WASPAA, pp. 125-129, Oct. 2019.

    Parameters
    ----------
    x: array_like (n_samples, n_ch)
        observed signal
    frlen: int, optional
        frame length of STFT analysis (default: # of sample)
    frsft: int, optional
        frame shift of STFT analysis (default: half of frlen)
    n_iter: int, optional
        The number of iterations
    phat: bool, optional
        If True, the PHAT is used as the weight function
    t0: float, optional
        The initial estimate of the time delay.
        If None, GCC with quadratic interpolation is used.
    ret_all: bool, optional
        If True, return the estimates at all iterations,
        else return only the final estimate

    Return
    ----------
    The sub-sample precision time delay estimate
    """

    # Check error and get parameters
    n_samples, n_ch = x.shape
    if n_samples < n_ch:
        x = x.T
        n_samples, n_ch = x.shape

    if n_ch != 2:
        raise Exception("Input channel must be two.")

    if frlen is None:
        frlen = n_samples

    if frsft is None:
        frsft = frlen // 2

    # STFT
    wnd = np.ones(frlen)
    X = mSTFT(x, frlen, frsft, wnd, zp=False)
    n_ch, n_frame, n_freq = X.shape
    w = 2.0 * np.pi * np.arange(0, n_freq) / frlen

    # compute weight function
    if phat:
        X /= np.abs(X) + 1e-15

    # compute cross spectrum and convert it to the sum of cosines
    xspec = np.mean(X[0, :, :] * np.conj(X[1, :, :]), 0)
    A = np.abs(xspec)
    phi = np.angle(xspec / A)
    A /= frlen
    A[1:-1] *= 2
    Aw = A * w
    Aw2 = A * w ** 2

    # initialization
    t = np.zeros([n_iter + 1])
    PHI = np.zeros([n_iter + 1])
    if t0 is None:
        t[0] = tdoa.GCC_with_parafit(xspec)
    else:
        t[0] = t0
    PHI[0] = cost_function(A, phi, t[0])

    # main
    for iter in range(1, n_iter + 1):
        # update phase estimate (auxiliary variables)
        theta = w * t[iter - 1] + phi

        # round to within 2pi
        theta -= np.round(theta / (2 * np.pi)) * 2 * np.pi

        # update time delay estimate
        k1 = np.sum(Aw * np.sin(theta))
        k2 = np.sum(Aw2 * np.sinc(theta / np.pi))
        t[iter] = t[iter - 1] - k1 / k2

        # compute the value of cost function
        PHI[iter] = cost_function(A, phi, t[iter])

    return t if ret_all else t[:, -1]


def cost_function(A, phi, t):
    """
    Compute the value of cost function by eq. (3)
    """
    N = (A.shape[0] - 1) * 2
    w = 2.0 * np.pi * np.arange(0, A.shape[0]) / N

    return np.sum(A * np.cos(w * t * phi)) / N
