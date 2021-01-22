# -*- coding: utf-8 -*-
import numpy as np
import maux_tdoa
from functions.tictoc import tictoc


if __name__ == "__main__":
    np.random.seed(577)

    # parameters
    l_sig = 2 ** 16
    frlen = l_sig
    n_iter = 30
    n_ch = 8
    true_tdoa = 3 * np.random.random(n_ch - 1)
    true_tdoa = np.append(0, true_tdoa)

    # simulation
    x = np.zeros([l_sig, n_ch])
    x[:, 0] = np.random.randn(l_sig)
    x1spec = np.fft.rfft(x[:, 0])
    freq = np.arange(0, l_sig // 2 + 1)
    w = 2 * np.pi * freq / l_sig

    # delayed signal
    for ch in range(1, n_ch):
        tmp = np.fft.rfft(x[:, 0]) * np.exp(1j * w * true_tdoa[ch])
        x[:, ch] = np.fft.irfft(tmp)

    # main
    est_maux = maux_tdoa.maux_tdoa(x)

    # deal with the minus time delay
    mask = est_maux > (l_sig / 2)
    est_maux -= l_sig * mask

    print("----------------------------")
    print("true_tdoa: {}\n est_tdoa: {}\n".format(true_tdoa, est_maux))

    print("----------------------------")
    err_maux = np.abs(est_maux - true_tdoa)
    print(f"error: {err_maux}\n")

    print("----------------------------")
    rmse_maux = np.sqrt(np.mean(err_maux ** 2))
    print(f"rmse: {rmse_maux}")
