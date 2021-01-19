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
    n_ch = 2
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
    est_naiv = maux_tdoa.init_tau(x, is_naive=True)
    est_para = maux_tdoa.init_tau(x)

    saux = tictoc("saux")
    saux.tic()
    est_saux = maux_tdoa.simple_maux_tdoa(x, frlen, n_iter=n_iter, ret_all=True)
    saux.toc()

    maux = tictoc("maux")
    maux.tic()
    est_maux = maux_tdoa.maux_tdoa(x, frlen, n_iter=n_iter, ret_all=True)
    maux.toc()

    # deal with the minus time delay
    mask = est_naiv > (l_sig / 2)
    est_naiv -= l_sig * mask
    mask = est_para > (l_sig / 2)
    est_para -= l_sig * mask
    mask = est_saux > (l_sig / 2)
    est_saux -= l_sig * mask
    mask = est_maux > (l_sig / 2)
    est_maux -= l_sig * mask

    print("----------------------------")
    print(
        "true_tdoa: {}\n est_naiv: {}\n est_para: {}\n est_saux: {}\n est_maux: {}\n".format(
            true_tdoa[1], est_naiv[1], est_para[1], est_saux[1, -1], est_maux[1, -1]
        )
    )
    print("----------------------------")

    err_naiv = np.abs(est_naiv[1] - true_tdoa[1])
    err_para = np.abs(est_para[1] - true_tdoa[1])
    err_saux = np.abs(est_saux[1, -1] - true_tdoa[1])
    err_maux = np.abs(est_maux[1, -1] - true_tdoa[1])
    print(
        "err_naiv: {}\nerr_para: {}\nerr_saux: {}\nerr_maux: {}".format(
            err_naiv, err_para, err_saux, err_maux
        )
    )
