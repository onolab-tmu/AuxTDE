# -*- coding: utf-8 -*-
import numpy as np
from functions.STFT import mSTFT
import tdoa
import aux_tdoa
from scipy.linalg import eigh


def maux_tdoa(
    x, frlen=None, frsft=None, n_iter=10, n_epoch=3, tau0=None, ret_all=False, tt=None
):
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

    # est tau main
    # initialization
    theta = np.zeros([n_ch, n_ch, n_freq])
    B = np.zeros([n_ch, n_ch, n_freq])
    C = np.zeros([n_ch, n_ch, n_freq])
    c = np.zeros([n_ch, n_freq])
    fixed_a = np.ones([n_ch, 1, n_freq])

    # initialization of time delays
    tau = np.zeros([n_ch, n_iter + 1])
    if tau0 is None:
        tau[:, -1] = init_tau(x)
    else:
        tau[:, -1] = tau0

    for epoch in range(n_epoch):
        tau[:, 0] = tau[:, -1]
        tdiff = tau[:, 0, np.newaxis].T - tau[:, 0, np.newaxis]

        # compute cost function
        cost = np.zeros([n_iter + 1])
        cost[0] = cost_function(tdiff, A, phi, w)

        # iterative updates
        for iter in range(1, n_iter + 1):
            # update phase estimates (auxiliary variables)
            theta = w[np.newaxis, np.newaxis, :] * tdiff[:, :, np.newaxis] + phi

            # round to within 2pi
            theta -= np.round(theta / (2 * np.pi)) * 2 * np.pi

            # update time delay estimates
            tmp = np.zeros([n_ch, n_ch, n_freq])
            for k in range(n_freq):
                tmp[:, :, k] = np.real(fixed_a[:, :, k]) @ np.real(fixed_a[:, :, k].T)
            B = tmp * A * np.sinc(theta / np.pi)
            C = w2 * (
                np.identity(n_ch)[:, :, np.newaxis]
                * np.sum(B, axis=0)[:, np.newaxis, :]
                - B
            )
            c = w * np.sum(B * theta, axis=0)

            tau[1:, iter] = tau[1:, iter - 1] - np.linalg.inv(
                np.sum(C[1:, 1:, :], axis=2)
            ) @ (np.sum(c[1:, :], axis=1))
            tdiff = tau[:, iter, np.newaxis].T - tau[:, iter, np.newaxis]

            # store the cost function
            cost[iter] = cost_function(tdiff, A, phi, w)

        fixed_tau = tau[:, -1]

        # a main
        tdiff = fixed_tau[:, np.newaxis].T - fixed_tau[:, np.newaxis]
        theta = w[np.newaxis, np.newaxis, :] * tdiff[:, :, np.newaxis] + phi
        Vp = np.abs(V) * np.exp(-1 * 1j * theta)
        for k in range(n_freq):
            _eig_val, eig_vec = eigh(
                np.real(Vp[:, :, k]), subset_by_index=[n_ch - 1, n_ch - 1]
            )
            fixed_a[:, :, k] = eig_vec / eig_vec[0, 0]

    return tau if ret_all else tau[:, -1]


def calc_SCM(X):
    X = X.transpose(2, 0, 1)
    V = X @ X.conj().swapaxes(1, 2)
    return V.transpose(1, 2, 0) / X.shape[1]


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


def cost_function(tdiff, A, phi, omega):
    return np.sum(
        A * np.cos(omega[np.newaxis, np.newaxis, :] * tdiff[:, :, np.newaxis] + phi)
    )


def auxiliary_function(tdiff, tdiff_0, A, phi, omega):
    tdiff = tdiff[:, :, np.newaxis]
    tdiff_0 = tdiff_0[:, :, np.newaxis]
    omega = omega[np.newaxis, np.newaxis, :]

    # phase estimates (auxiliary variables)
    theta = omega * tdiff_0 + phi

    # round to within 2pi
    n = np.round(theta / (2 * np.pi))
    theta -= 2 * n * np.pi

    Q = np.sum(
        A
        * (
            (-0.5 * np.sinc(theta / np.pi)) * (omega * tdiff + phi - 2 * n * np.pi) ** 2
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
