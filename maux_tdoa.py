# -*- coding: utf-8 -*-
import numpy as np
from functions.STFT import mSTFT
import tdoa
import aux_tdoa


def maux_tdoa(x, frlen=None, frsft=None, n_iter=10, tau0=None, ret_all=False):
    """
    Auxilliary function based sub-sample time delays estimation.

    parameters
    ----------
    x: array_like (n_sample, n_ch)
        Multichannel observations
    frlen: float, optional
        FFT frame length for STFT (default: n_sample)
    frsft: float, optional
        Proportion of frame shift (default: half overlap)
    n_iter: int, optional
        The number of iterations
    tau0: float, optional
        The initial estimates of the time delay.
        If None, GCC with quadratic interpolation is used.
    ret_all: bool, optional
        If True, return the estimates at all iterations,
        else return only the final estimates

    return
    ----------
    The sub-sample precision time delay estimates
    """

    # Check error and get parameters
    n_samples, n_ch = x.shape
    if n_samples < n_ch:
        x = x.T
        n_samples, n_ch = x.shape

    if frlen is None:
        frlen = n_samples

    if frsft is None:
        frsft = frlen // 2

    # STFT
    wnd = np.ones(frlen)
    X = mSTFT(x, frlen, frsft, wnd, zp=False)
    n_ch, n_frame, n_freq = X.shape

    # compute variables/parameters
    w = 2.0 * np.pi * np.arange(0, n_freq) / frlen
    w2 = w ** 2
    V = calc_SCM(X)
    A = np.abs(V)
    phi = np.angle(V / A)
    A /= frlen
    A[:, :, 1:-1] *= 2

    # main
    # initialization
    theta = np.zeros([n_ch, n_ch, n_freq])
    B = np.zeros([n_ch, n_ch, n_freq])
    C = np.zeros([n_ch, n_ch, n_freq])
    c = np.zeros([n_ch, n_freq])

    # initialization of time delays
    tau = np.zeros([n_ch, n_iter + 1])
    if tau0 is None:
        tau[:, 0] = init_tau(x)
    else:
        tau[:, 0] = tau0
    tdiff = tau[:, 0, np.newaxis].T - tau[:, 0, np.newaxis]

    # compute cost function
    cost = np.zeros([n_iter + 1])
    cost[0] = cost_function(tdiff, A, phi, w)

    # iterative updates
    for iter in range(0, n_iter):
        # update phase estimates (auxiliary variables)
        theta = w[np.newaxis, np.newaxis, :] * tdiff[:, :, np.newaxis] + phi

        # round to within 2pi
        theta -= np.round(theta / (2 * np.pi)) * 2 * np.pi

        # update time delay estimates
        B = A * np.sinc(theta / np.pi)
        C = w2 * (
            np.identity(n_ch)[:, :, np.newaxis] * np.sum(B, axis=0)[:, np.newaxis, :]
            - B
        )
        c = w * np.sum(B * theta, axis=0)

        tau[1:, iter + 1] = tau[1:, iter] - np.linalg.inv(
            np.sum(C[1:, 1:, :], axis=2)
        ) @ (np.sum(c[1:, :], axis=1))
        tdiff = tau[:, iter + 1, np.newaxis].T - tau[:, iter + 1, np.newaxis]

        # store the cost function
        cost[iter + 1] = cost_function(tdiff, A, phi, w)

    # tdiff = tau[:, -1, np.newaxis].T - tau[:, -1, np.newaxis]
    # cost[-1] = cost_function(tdiff, A, phi, w)

    return tau if ret_all else tau[:, -1]


def calc_SCM(X):
    n_ch, n_frame, n_freq = X.shape
    V = np.zeros([n_ch, n_ch, n_freq], dtype=complex)
    for f in range(n_freq):
        V[:, :, f] = X[:, :, f] @ X[:, :, f].conj().T
    return V / n_frame


def init_tau(x, is_naive=False):
    # initialization
    n_samples, n_ch = x.shape
    tau = np.zeros([n_ch])
    tau_naive = np.zeros([n_ch])

    # compute cross-spectrum
    X1 = np.fft.rfft(x[:, 0])
    for ch in range(1, n_ch):
        X2 = np.fft.rfft(x[:, ch])

        # compute time delay estimates
        tau[ch], tau_naive[ch] = tdoa.GCC_with_parafit(X1 * np.conj(X2), ret_GCC=True)

    return tau_naive if is_naive else tau


def cost_function(tdiff, A, phi, omega):
    return np.sum(
        A * np.cos(omega[np.newaxis, np.newaxis, :] * tdiff[:, :, np.newaxis] + phi)
    )


def simple_maux_tdoa(x, frlen=None, n_iter=10, ret_all=False):
    n_samples, n_ch = x.shape
    tau = np.zeros([n_ch, n_iter + 1])
    x_2ch = np.zeros([n_samples, 2])
    x_2ch[:, 0] = x[:, 0]
    for ch in range(1, n_ch):
        x_2ch[:, 1] = x[:, ch]
        tau[ch, :] = aux_tdoa.aux_tdoa(x_2ch, frlen, n_iter=n_iter, ret_all=True)
    return tau if ret_all else tau[:, -1]