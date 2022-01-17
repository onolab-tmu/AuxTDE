# -*- coding: utf-8 -*-
import numpy as np

import single_AuxTDE
import TDE
from functions.STFT import mSTFT


def AuxTDE(
    x,
    frlen=None,
    frsft=None,
    n_iter_t=5,
    n_iter_a=5,
    n_epoch=2,
    a0=None,
    tau0=None,
    average=True,
):
    """
    Auxilliary function based estimation of sub-sample time delays.

    parameters
    ----------
    x: array_like (n_sample, n_ch)
        Multichannel observations
    frlen: float, optional
        FFT frame length for STFT (default: n_sample)
    frsft: float, optional
        Proportion of frame shift (default: half overlap)
    n_iter_t: int, optional
        The number of iterations for tau updates
    n_iter_a: int, optional
        The number of iterations for a updates
    n_epoch: int, optional
        The number of epochs for alternating updates of tau and a
    a0: float, optional
        The initial estimates of the amplitude.
        If None, 1 is used for all frequencies.
    tau0: float, optional
        The initial estimates of the time delays.
        If None, GCC with quadratic interpolation is used.
    average: bool, optional
        If true, frequency-independent amplitude is estimated.

    return
    ----------
    tau: array_like (n_iter_t + 1, n_ch)
        The sub-sample precision time delay estimates
        tau[0, :] are the initial estimates.
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
    tau = init_tau(x) if tau0 is None else tau0
    a = np.ones([n_freq, n_ch, 1]) / np.sqrt(n_ch) if a0 is None else a0
    sigma = np.ones([n_ch])
    f_update_a = update_a_mean if average else update_a_freq

    # alternating updates
    for epoch in range(n_epoch):
        weighted_A = A / (sigma[:, np.newaxis] @ sigma[np.newaxis, :])

        # time delay estimation
        for iter in range(n_iter_t):
            tau[1:] = update_tau(a, tau, w, A, phi)

        # amplitude estimation
        for iter in range(n_iter_a):
            a = f_update_a(a, tau, w, A, phi)

        # update sigma
        sigma = update_sigma(a, tau, X, w)

    # deal with the minus time delay
    mask = tau > (n_samples / 2)
    tau -= n_samples * mask

    return tau


def update_tau(a, tau, w, A, phi):
    w2 = w ** 2
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    n_ch = tau.shape[0]

    # update phase estimates (auxiliary variables)
    theta = w[:, np.newaxis, np.newaxis] * tdiff[np.newaxis, :, :] + phi

    # round to within 2pi
    theta -= np.round(theta / (2 * np.pi)) * 2 * np.pi

    # update coefficients of auxiliary function
    B = a @ a.swapaxes(1, 2) * A * np.sinc(theta / np.pi)
    C = w2[:, np.newaxis, np.newaxis] * (
        np.identity(n_ch)[np.newaxis, :, :] * np.sum(B, axis=1)[:, np.newaxis, :] - B
    )
    c = w[:, np.newaxis] * np.sum(B * theta, axis=1)

    # update time delay estimates
    retval = tau[1:] - np.linalg.inv(np.sum(C[:, 1:, 1:], axis=0)) @ (
        np.sum(c[:, 1:], axis=0)
    )

    return retval


def update_a_freq(a, tau, w, A, phi):
    # update auxiliary variables
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    theta = w[:, np.newaxis, np.newaxis] * tdiff[np.newaxis, :, :] + phi

    # update amplitudes
    V_prime = np.real(A * np.exp(-1 * 1j * theta))
    a = max_and_norm(V_prime @ a)

    return a


def update_a_mean(a, tau, w, A, phi):
    # update auxiliary variables
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    theta = w[:, np.newaxis, np.newaxis] * tdiff[np.newaxis, :, :] + phi

    # update amplitudes
    V_prime = np.mean(np.real(A * np.exp(-1 * 1j * theta)), axis=0)
    a = max_and_norm(V_prime @ a)

    return a


def max_and_norm(vec):
    vec = np.maximum(vec, 1e-10)
    return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis, :]


def update_sigma(a, tau, X, w):
    g = calcRTF(a, tau, w)
    tmp = X - g * (g.conj().swapaxes(1, 2) @ X) / (g.conj().swapaxes(1, 2) @ g)
    tmp2 = np.real(tmp.conj() * tmp)
    tmp2[1:-1, :, :] *= 2
    sigma2 = np.mean(tmp2, axis=(0, 2))

    return np.sqrt(sigma2)


def calcRTF(a, tau, w):
    return a * np.exp(
        -1j * w[:, np.newaxis, np.newaxis] * tau[np.newaxis, :, np.newaxis]
    )


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
        tau[ch], tau_naive[ch] = TDE.GCC_with_parafit(
            X[:, 0] * np.conj(X[:, ch]), ret_GCC=True
        )

    return tau_naive if is_naive else tau


def cost_function(a, tau, A, phi, w):
    amat = a * a.swapaxes(1, 2)
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    cost = np.sum(
        amat * A * np.cos(w[:, np.newaxis, np.newaxis] * tdiff[np.newaxis, :, :] + phi)
    )
    return cost


def cost_function_mat(a, V):
    cost = np.sum(a.swapaxes(1, 2) @ V @ a)
    return cost


def auxiliary_function(a, tau, init_tau, A, phi, w):
    amat = a * a.swapaxes(1, 2)
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    tdiff = tdiff[np.newaxis, :, :]
    tdiff_0 = init_tau[:, np.newaxis].T - init_tau[:, np.newaxis]
    tdiff_0 = tdiff_0[np.newaxis, :, :]
    w = w[:, np.newaxis, np.newaxis]

    # phase estimates (auxiliary variables)
    theta = w * tdiff_0 + phi

    # round to within 2pi
    n = np.round(theta / (2 * np.pi))
    theta -= 2 * n * np.pi

    # main
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


def PW_AuxTDE(x, frlen=None, n_iter=10, ret_all=False):
    n_samples, n_ch = x.shape
    tau = np.zeros([n_ch, n_iter + 1])
    x_2ch = np.zeros([n_samples, 2])
    x_2ch[:, 0] = x[:, 0]
    for ch in range(1, n_ch):
        x_2ch[:, 1] = x[:, ch]
        tau[ch, :] = single_AuxTDE.AuxTDE(x_2ch, frlen, n_iter=n_iter, ret_all=True)
    return tau if ret_all else tau[:, -1]
