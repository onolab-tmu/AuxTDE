# -*- coding: utf-8 -*-
import numpy as np

import AuxTDE

if __name__ == "__main__":
    np.random.seed(577)

    # signal parameters
    l_sig = 2 ** 14
    n_ch = 8
    true_TD = 5 * np.random.random(n_ch - 1)
    true_TD = np.append(0, true_TD)

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
    tau = AuxTDE.AuxTDE(x)

    # result
    print("----------------------------")
    print("true_TD: {}\n est_TD: {}".format(true_TD, tau))

    print("----------------------------")
    err = np.abs(tau - true_TD)
    print(f"error: {err}")

    print("----------------------------")
    rmse = np.sqrt(np.mean(err ** 2))
    print(f"rmse: {rmse}")
