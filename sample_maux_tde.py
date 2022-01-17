# -*- coding: utf-8 -*-
import numpy as np

import maux_tde
from functions.tictoc import tictoc

if __name__ == "__main__":
    np.random.seed(577)

    # signal parameters
    l_sig = 2 ** 12
    n_ch = 8
    true_tdoa = 5 * np.random.random(n_ch - 1)
    true_tdoa = np.append(0, true_tdoa)

    # experimental parameters
    n_iter_t = 10
    n_iter_a = 10
    n_epoch = 3

    # simulation
    x = np.zeros([l_sig, n_ch])
    x[:, 0] = np.random.randn(l_sig)
    x1spec = np.fft.rfft(x[:, 0])
    freq = np.arange(0, l_sig // 2 + 1)
    w = 2 * np.pi * freq / l_sig

    # delayed signals
    amp = np.random.random([l_sig // 2 + 1, n_ch]) + 0.5
    for ch in range(1, n_ch):
        tmp = amp[:, ch] * np.fft.rfft(x[:, 0]) * np.exp(1j * w * true_tdoa[ch])
        x[:, ch] = np.fft.irfft(tmp)

    # main
    est_naiv = maux_tde.init_tau(x, is_naive=True)
    est_para = maux_tde.init_tau(x)

    saux = tictoc("saux")
    saux.tic()
    est_saux = maux_tde.simple_maux_tde(x, n_iter=n_iter_t)
    saux.toc()

    maux = tictoc("maux")
    maux.tic()
    est_maux = maux_tde.maux_tde(x, n_iter_t=n_iter_t, n_iter_a=1, n_epoch=1)
    maux.toc()

    maux_amp = tictoc("maux_amp")
    maux_amp.tic()
    est_maux_amp = maux_tde.maux_tde(
        x, n_iter_t=n_iter_t, n_iter_a=n_iter_a, n_epoch=n_epoch, average=False
    )
    maux_amp.toc()

    maux_mamp = tictoc("maux_mamp")
    maux_mamp.tic()
    est_maux_mamp = maux_tde.maux_tde(
        x, n_iter_t=n_iter_t, n_iter_a=n_iter_a, n_epoch=n_epoch, average=True
    )
    maux_mamp.toc()

    # results
    print("----------------------------")
    print(
        (
            "true_tdoa: {}\n"
            "est_naiv: {}\n"
            "est_para: {}\n"
            "est_saux: {}\n"
            "est_maux: {}\n"
            "est_maux_amp: {}\n"
            "est_maux_mamp: {}"
        ).format(
            true_tdoa,
            est_naiv,
            est_para,
            est_saux,
            est_maux,
            est_maux_amp,
            est_maux_mamp,
        )
    )
    print("----------------------------")

    err_naiv = np.abs(est_naiv - true_tdoa)
    err_para = np.abs(est_para - true_tdoa)
    err_saux = np.abs(est_saux - true_tdoa)
    err_maux = np.abs(est_maux - true_tdoa)
    err_maux_amp = np.abs(est_maux_amp - true_tdoa)
    err_maux_mamp = np.abs(est_maux_mamp - true_tdoa)
    print(
        (
            "err_naiv: {}\n"
            "err_para: {}\n"
            "err_saux: {}\n"
            "err_maux: {}\n"
            "err_maux_amp: {}\n"
            "err_maux_mamp: {}"
        ).format(err_naiv, err_para, err_saux, err_maux, err_maux_amp, err_maux_mamp)
    )

    print("----------------------------")
    rmse_naiv = np.sqrt(np.mean(err_naiv ** 2))
    rmse_para = np.sqrt(np.mean(err_para ** 2))
    rmse_saux = np.sqrt(np.mean(err_saux ** 2))
    rmse_maux = np.sqrt(np.mean(err_maux ** 2))
    rmse_maux_amp = np.sqrt(np.mean(err_maux_amp ** 2))
    rmse_maux_mamp = np.sqrt(np.mean(err_maux_mamp ** 2))
    print(
        (
            "rmse_naiv     : {}\n"
            "rmse_para     : {}\n"
            "rmse_saux     : {}\n"
            "rmse_maux     : {}\n"
            "rmse_maux_amp : {}\n"
            "rmse_maux_mamp: {}"
        ).format(
            rmse_naiv, rmse_para, rmse_saux, rmse_maux, rmse_maux_amp, rmse_maux_mamp
        )
    )
