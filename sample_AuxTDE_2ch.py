# -*- coding: utf-8 -*-
import numpy as np

import AuxTDE
from functions.tictoc import tictoc

if __name__ == "__main__":
    np.random.seed(577)

    # signal parameters
    l_sig = 2 ** 14
    n_ch = 2
    true_TD = 5 * np.random.random(n_ch - 1)
    true_TD = np.append(0, true_TD)

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
        tmp = amp[:, ch] * np.fft.rfft(x[:, 0]) * np.exp(1j * w * true_TD[ch])
        x[:, ch] = np.fft.irfft(tmp)

    # main
    gcc = tictoc("PW-GCC")
    gcc.tic()
    est_gcc = AuxTDE.init_tau(x, is_naive=True)
    gcc.toc()

    parafit = tictoc("PW-Parafit")
    parafit.tic()
    est_parafit = AuxTDE.init_tau(x)
    parafit.toc()

    pw_aux = tictoc("PW-AuxTDE")
    pw_aux.tic()
    est_pw_aux = AuxTDE.PW_AuxTDE(x, n_iter=n_iter_t)
    pw_aux.toc()

    aux_unit = tictoc("AuxTDE_unitAmp")
    aux_unit.tic()
    est_aux_unit = AuxTDE.AuxTDE(x, n_iter_t=n_iter_t, n_iter_a=1, n_epoch=1)
    aux_unit.toc()

    aux_freq = tictoc("AuxTDE_freqAmp")
    aux_freq.tic()
    est_aux_freq = AuxTDE.AuxTDE(
        x, n_iter_t=n_iter_t, n_iter_a=n_iter_a, n_epoch=n_epoch, average=False
    )
    aux_freq.toc()

    aux_shrd = tictoc("AuxTDE_shrdAmp")
    aux_shrd.tic()
    est_aux_shrd = AuxTDE.AuxTDE(
        x, n_iter_t=n_iter_t, n_iter_a=n_iter_a, n_epoch=n_epoch, average=True
    )
    aux_shrd.toc()

    # results
    print("----------------------------")
    print(
        (
            "true_TD: \n{}\n"
            "est_GCC: \n{}\n"
            "est_Parafit: \n{}\n"
            "est_PW-AuxTDE: \n{}\n"
            "est_AuxTDE_unitAmp: \n{}\n"
            "est_AuxTDE_freqAmp: \n{}\n"
            "est_AuxTDE_shrdAmp: \n{}"
        ).format(
            true_TD,
            est_gcc,
            est_parafit,
            est_pw_aux,
            est_aux_unit,
            est_aux_freq,
            est_aux_shrd,
        )
    )
    print("----------------------------")

    err_gcc = np.abs(est_gcc - true_TD)
    err_parafit = np.abs(est_parafit - true_TD)
    err_pw_aux = np.abs(est_pw_aux - true_TD)
    err_aux_unit = np.abs(est_aux_unit - true_TD)
    err_aux_freq = np.abs(est_aux_freq - true_TD)
    err_aux_shrd = np.abs(est_aux_shrd - true_TD)
    print(
        (
            "err_GCC: \n{}\n"
            "err_Parafit: \n{}\n"
            "err_PW-AuxTDE: \n{}\n"
            "err_AuxTDE_unitAmp: \n{}\n"
            "err_AuxTDE_freqAmp: \n{}\n"
            "err_AuxTDE_shrdAmp: \n{}"
        ).format(
            err_gcc, err_parafit, err_pw_aux, err_aux_unit, err_aux_freq, err_aux_shrd
        )
    )

    print("----------------------------")
    rmse_gcc = np.sqrt(np.mean(err_gcc ** 2))
    rmse_parafit = np.sqrt(np.mean(err_parafit ** 2))
    rmse_pw_aux = np.sqrt(np.mean(err_pw_aux ** 2))
    rmse_aux_unit = np.sqrt(np.mean(err_aux_unit ** 2))
    rmse_aux_freq = np.sqrt(np.mean(err_aux_freq ** 2))
    rmse_aux_shrd = np.sqrt(np.mean(err_aux_shrd ** 2))
    print(
        (
            "RMSE_GCC:            {}\n"
            "RMSE_Parafit:        {}\n"
            "RMSE_PW-AuxTDE:      {}\n"
            "RMSE_AuxTDE_unitAmp: {}\n"
            "RMSE_AuxTDE_freqAmp: {}\n"
            "RMSE_AuxTDE_shrdAmp: {}"
        ).format(
            rmse_gcc,
            rmse_parafit,
            rmse_pw_aux,
            rmse_aux_unit,
            rmse_aux_freq,
            rmse_aux_shrd,
        )
    )
