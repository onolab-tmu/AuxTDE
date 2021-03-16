# -*- coding: utf-8 -*-
import numpy as np
from functions.STFT import mSTFT
import tdoa
import aux_tdoa


def maux_tdoa(x, frlen=None, frsft=None, n_iter=10, n_epoch=3, tau0=None):
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
    X = mSTFT(x, frlen, frsft, wnd, zp=False).transpose(2, 0, 1)
    n_freq, n_ch, n_frame = X.shape

    # compute variables/parameters
    w = 2.0 * np.pi * np.arange(0, n_freq) / frlen
    V = calc_SCM(X)
    A = np.abs(V)
    phi = np.angle(V / A)
    A /= frlen
    A[1:-1, :, :] *= 2

    ## main
    # initialization
    a = np.ones([n_freq, n_ch, 1])
    tau = np.zeros([n_iter + 1, n_ch])
    if tau0 is None:
        tau[-1, :] = init_tau(x)
    else:
        tau[-1, :] = tau0

    # main
    for epoch in range(n_epoch):
        # iterative updates for time delay estimation
        tau[0, :] = tau[-1, :]
        for iter in range(1, n_iter + 1):
            tau[iter, 1:] = update_tau(a, tau[iter - 1, :], w, A, phi)

        # amplitude estimation
        a = update_a(tau[-1, :], w, A, phi)

    return a, tau[-1, :]


def maux_tdoa_retall(x, frlen=None, frsft=None, n_iter=10, n_epoch=3, tau0=None):
    """
    maux_tdoa for analysis purpose

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
    X = mSTFT(x, frlen, frsft, wnd, zp=False).transpose(2, 0, 1)
    n_freq, n_ch, n_frame = X.shape

    # compute variables/parameters
    w = 2.0 * np.pi * np.arange(0, n_freq) / frlen
    V = calc_SCM(X)
    A = np.abs(V)
    phi = np.angle(V / A)
    A /= frlen
    A[1:-1, :, :] *= 2

    ## main
    # initialization
    a = np.ones([n_epoch + 1, n_freq, n_ch, 1])
    tau = np.zeros([n_epoch + 1, n_iter + 1, n_ch])
    cost = np.zeros([n_epoch + 1, n_iter + 1])
    if tau0 is None:
        tau[0, -1, :] = init_tau(x)
    else:
        tau[0, -1, :] = tau0

    # main
    for epoch in range(1, n_epoch + 1):
        # iterative updates for time delay estimation
        tau[epoch, 0, :] = tau[epoch - 1, -1, :]
        cost[epoch, 0] = cost_function(a[epoch, :, :, :], tau[0, -1, :], A, phi, w)
        for iter in range(1, n_iter + 1):
            tau[epoch, iter, 1:] = update_tau(
                a[epoch - 1, :, :, :], tau[epoch, iter - 1, :], w, A, phi
            )
            cost[epoch, iter] = cost_function(
                a[epoch, :, :, :], tau[epoch, iter, :], A, phi, w
            )

        # amplitude estimation
        a[epoch, :, :, :] = update_a(tau[epoch, -1, :], w, A, phi)
        cost[epoch, -1] = cost_function(a[epoch, :, :, :], tau[epoch, -1, :], A, phi, w)

    return a, tau, cost


def update_a(tau, w, A, phi):
    # update auxiliary variables
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    theta = w[:, np.newaxis, np.newaxis] * tdiff[np.newaxis, :, :] + phi

    # solve eigenvalue decomposition
    _eig_val, eig_vec = np.linalg.eigh(np.real(A * np.exp(-1 * 1j * theta)))

    # normalization
    a = eig_vec[:, :, -1, np.newaxis] / eig_vec[:, 0, -1, np.newaxis, np.newaxis]

    return a


def update_tau(a, tau, w, A, phi):
    w2 = w ** 2
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    n_ch = tau.shape[0]

    # update phase estimates (auxiliary variables)
    theta = w[:, np.newaxis, np.newaxis] * tdiff[np.newaxis, :, :] + phi

    # round to within 2pi
    theta -= np.round(theta / (2 * np.pi)) * 2 * np.pi

    # update time delay estimates
    B = a @ a.swapaxes(1, 2) * A * np.sinc(theta / np.pi)
    C = w2[:, np.newaxis, np.newaxis] * (
        np.identity(n_ch)[np.newaxis, :, :] * np.sum(B, axis=1)[:, np.newaxis, :] - B
    )
    c = w[:, np.newaxis] * np.sum(B * theta, axis=1)

    retval = tau[1:] - np.linalg.inv(np.sum(C[:, 1:, 1:], axis=0)) @ (
        np.sum(c[:, 1:], axis=0)
    )

    return retval


def calc_SCM(X):
    # compute spatial covariance matrix
    V = X @ X.conj().swapaxes(1, 2)
    return V / X.shape[2]


def init_tau(x, is_naive=False):
    # initialization
    n_samples, n_ch = x.shape
    tau = np.zeros([n_ch])
    tau_naive = np.zeros([n_ch])

    # compute cross-spectrum
    X = np.fft.rfft(x, axis=0)

    # compute time delay estimates
    for ch in range(1, n_ch):
        tau[ch], tau_naive[ch] = tdoa.GCC_with_parafit(
            X[:, 0] * np.conj(X[:, ch]), ret_GCC=True
        )

    return tau_naive if is_naive else tau


def cost_function(a, tau, A, phi, w):
    amat = a * a.swapaxes(1, 2)
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    return np.sum(
        amat * A * np.cos(w[:, np.newaxis, np.newaxis] * tdiff[np.newaxis, :, :] + phi)
    )


def auxiliary_function(a, tau, init_tau, A, phi, w):
    amat = a * a.swapaxes(1, 2)
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    tdiff_0 = init_tau[:, np.newaxis].T - init_tau[:, np.newaxis]
    tdiff = tdiff[np.newaxis, :, :]
    tdiff_0 = tdiff_0[np.newaxis, :, :]
    w = w[:, np.newaxis, np.newaxis]

    # phase estimates (auxiliary variables)
    theta = w * tdiff_0 + phi

    # round to within 2pi
    n = np.round(theta / (2 * np.pi))
    theta -= 2 * n * np.pi

    Q = np.sum(
        amat
        * A
        * (
            (-0.5 * np.sinc(theta / np.pi)) * (w * tdiff + phi - 2 * n * np.pi) ** 2
            + np.cos(theta)
            + theta * np.sin(theta) / 2
        )
    )

    return Q


def simple_maux_tdoa(x, frlen=None, n_iter=10, ret_all=False):
    n_samples, n_ch = x.shape
    tau = np.zeros([n_ch, n_iter + 1])
    x_2ch = np.zeros([n_samples, 2])
    x_2ch[:, 0] = x[:, 0]
    for ch in range(1, n_ch):
        x_2ch[:, 1] = x[:, ch]
        tau[ch, :] = aux_tdoa.aux_tdoa(x_2ch, frlen, n_iter=n_iter, ret_all=True)
    return tau if ret_all else tau[:, -1]
