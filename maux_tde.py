# -*- coding: utf-8 -*-
import numpy as np

import aux_tde
import tde
from functions.STFT import mSTFT


def maux_tde(
    x,
    frlen=None,
    frsft=None,
    n_iter_t=10,
    n_iter_a=10,
    n_epoch=1,
    tau0=None,
    mean_a=False,
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
    n_iter_t: int, optional
        The number of iterations for tau updates
    n_iter_a: int, optional
        The number of iterations for a updates
    n_epoch: int, optional
        The number of epochs for alternating updates of tau and a
    tau0: float, optional
        The initial estimates of the time delay.
        If None, GCC with quadratic interpolation is used.
    mean_a: bool, optional
        If true, frequency-independent a is used.

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
    a = np.ones([n_freq, n_ch, 1]) / np.sqrt(n_ch)
    tau = np.zeros([n_iter_t + 1, n_ch])
    if tau0 is None:
        tau[-1, :] = init_tau(x)
    else:
        tau[-1, :] = tau0

    # alternating updates
    for epoch in range(n_epoch):
        # iterative updates for time delay estimation
        # use the last estimates of the previous epoch for initialization
        tau[0, :] = tau[-1, :]
        for iter in range(1, n_iter_t + 1):
            tau[iter, 1:] = update_tau(a, tau[iter - 1, :], w, A, phi)

        # amplitude estimation
        tmp, J = update_a(tau[-1, :], w, A, phi, n_iter_a, a, mean_a)
        a = tmp[-1, :, :, :]

    # deal with the minus time delay
    mask = tau > (n_samples / 2)
    tau -= n_samples * mask

    return tau[-1, :]


def maux_tde_retall(
    x,
    frlen=None,
    frsft=None,
    n_iter_t=10,
    n_iter_a=10,
    n_epoch=3,
    tau0=None,
    mean_a=False,
):
    """
    maux_tde for analysis purpose

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
    a = np.ones([n_epoch + 1, n_iter_a + 1, n_freq, n_ch, 1]) / np.sqrt(n_ch)
    tau = np.zeros([n_epoch + 1, n_iter_t + 1, n_ch])
    cost = np.zeros([n_epoch + 1, n_iter_t + n_iter_a + 2])
    if tau0 is None:
        tau[0, -1, :] = init_tau(x)
    else:
        tau[0, -1, :] = tau0
    cost[0, -1] = cost_function(a[0, 0, :, :, :], tau[0, -1, :], A, phi, w)
    print(tau[0, -1, :])

    # main
    for epoch in range(1, n_epoch + 1):
        # iterative updates for time delay estimation
        tau[epoch, 0, :] = tau[epoch - 1, -1, :]
        current_a = a[epoch - 1, -1, :, :, :]
        cost[epoch, 0] = cost_function(current_a, tau[epoch, 0, :], A, phi, w)
        for iter in range(1, n_iter_t + 1):
            tau[epoch, iter, 1:] = update_tau(
                current_a, tau[epoch, iter - 1, :], w, A, phi
            )
            cost[epoch, iter] = cost_function(current_a, tau[epoch, iter, :], A, phi, w)

        # amplitude estimation
        a[epoch, :, :, :, :], cost[epoch, n_iter_t + 1 :] = update_a(
            tau[epoch, -1, :],
            w,
            A,
            phi,
            n_iter_a,
            a0=a[epoch - 1, -1, :, :, :],
            mean_a=mean_a,
        )

    return a, tau, cost


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


def update_a(tau, w, A, phi, n_iter, a0=None, mean_a=False):
    # update auxiliary variables
    tdiff = tau[:, np.newaxis].T - tau[:, np.newaxis]
    theta = w[:, np.newaxis, np.newaxis] * tdiff[np.newaxis, :, :] + phi

    # update amplitude estimates
    if mean_a:
        a, J = mmeig_mean(np.real(A * np.exp(-1 * 1j * theta)), n_iter, a0)
    else:
        a, J = mmeig(np.real(A * np.exp(-1 * 1j * theta)), n_iter, a0)

    return a, J


def mmeig(V, n_iter, a0):
    n_freq, n_ch = V.shape[0:2]
    a = np.ones([n_iter + 1, n_freq, n_ch, 1])

    # initialization
    if a0 is None:
        _eig_val, eig_vec = np.linalg.eigh(V)
        tmp = eig_vec[:, :, -1, np.newaxis] / eig_vec[:, 0, -1, np.newaxis, np.newaxis]
        a[0, :, :, :] = max_and_norm(tmp)
    else:
        a[0, :, :, :] = a0

    # updates
    J = np.zeros(n_iter + 1)
    J[0] = cost_function_mat(a[0, :, :, :], V)
    for i in range(1, n_iter + 1):
        a[i, :, :, :] = max_and_norm(V @ a[i - 1, :, :, :])
        J[i] = cost_function_mat(a[i, :, :, :], V)

    return a, J


def mmeig_mean(V, n_iter, a0):
    n_freq, n_ch = V.shape[0:2]
    Vsum = np.sum(V, axis=0) / n_freq
    a = np.ones([n_iter + 1, n_ch, 1])

    # initialization
    if a0 is None:
        _eig_val, eig_vec = np.linalg.eigh(Vsum)
        tmp = eig_vec[:, -1, np.newaxis] / eig_vec[0, -1, np.newaxis, np.newaxis]
        a[0, :, :] = max_and_norm2(tmp)
    else:
        a[0, :, :] = a0[0, :, :]

    # updates
    J = np.zeros(n_iter + 1)
    J[0] = cost_function_mat(a[0, np.newaxis, :, :], V)
    for i in range(1, n_iter + 1):
        a[i, :, :] = max_and_norm2(Vsum @ a[i - 1, :, :])
        J[i] = cost_function_mat(a[i, np.newaxis, :, :], V)

    return a[:, np.newaxis, :, :], J


def max_and_norm(vec):
    vec = np.maximum(vec, 1e-10)
    return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis, :]


def max_and_norm2(vec):
    vec = np.maximum(vec, 1e-10)
    return vec / np.linalg.norm(vec, axis=0)[np.newaxis, :]


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
        tau[ch], tau_naive[ch] = tde.GCC_with_parafit(
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


def simple_maux_tde(x, frlen=None, n_iter=10, ret_all=False):
    n_samples, n_ch = x.shape
    tau = np.zeros([n_ch, n_iter + 1])
    x_2ch = np.zeros([n_samples, 2])
    x_2ch[:, 0] = x[:, 0]
    for ch in range(1, n_ch):
        x_2ch[:, 1] = x[:, ch]
        tau[ch, :] = aux_tde.aux_tde(x_2ch, frlen, n_iter=n_iter, ret_all=True)
    return tau if ret_all else tau[:, -1]
