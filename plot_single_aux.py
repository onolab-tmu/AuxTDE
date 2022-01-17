# -*- coding: utf-8 -*-
import argparse
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import AuxTDE
from functions.STFT import mSTFT


def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i", "--init_tau", help="Initial time delay estimates", type=str, default="1",
    )
    parser.add_argument(
        "-m", "--n_mesh", help="# of mesh for plot", type=int, default=300,
    )
    parser.add_argument(
        "-o", "--out_path", type=str, help="Path for save figure", default=None,
    )
    parser.add_argument(
        "-w",
        "--n_worker",
        help="# of workers for parallel processing",
        type=int,
        default=4,
    )
    return parser.parse_args()


def cost_function(args):
    x = args[0]
    tau = np.array([0, x])

    return AuxTDE.cost_function(a, tau, A, phi, w)


def auxiliary_function(args):
    x = args[0]
    tau = np.array([0, x])

    return AuxTDE.auxiliary_function(a, tau, init_tau, A, phi, w)


def mp_init():
    import mkl

    mkl.set_num_threads(1)


if __name__ == "__main__":
    np.random.seed(577)
    args = parse_cmd_line_arguments()

    global a
    global A
    global w
    global phi
    global init_tau

    # parameters
    l_sig = 2 ** 10
    frlen = l_sig
    frsft = frlen // 2
    n_iter = 30
    n_ch = 2
    true_tdoa = np.array([1.5, 3])
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

    # STFT
    wnd = np.ones(frlen)
    X = mSTFT(x, frlen, frsft, wnd, zp=False).transpose(2, 0, 1)
    n_freq, n_ch, n_frame = X.shape

    # compute variables/parameters
    w = 2.0 * np.pi * np.arange(0, n_freq) / frlen
    w2 = w ** 2
    V = AuxTDE.calc_SCM(X)
    A = np.abs(V)
    phi = np.angle(V / A)
    A /= frlen
    A[:, :, 1:-1] *= 2
    a = np.ones([n_freq, n_ch, 1])

    ## Objective function
    # set range
    tau = np.linspace(-3, 5, args.n_mesh)
    n_col = len(X)

    # compute objective function
    cost = np.zeros([n_col])
    with Pool(args.n_worker, initializer=mp_init) as p:
        cost = p.map(cost_function, list(zip(np.ravel(tau))))

    ## auxiliary function
    init_tau = np.array(np.append([0], float(args.init_tau)))

    # compute auxiliary function
    af = np.zeros([n_col])
    with Pool(args.n_worker, initializer=mp_init) as p:
        af = np.array(p.map(auxiliary_function, list(zip(np.ravel(tau)))))
    af = np.ma.masked_where(af < 0, af)
    mask = np.ma.masked_where(af >= 0, af)

    # plot
    plt.plot(tau, cost, color="#0000ff99")
    plt.plot(tau, af, color="#eb3929ff")
    plt.plot(tau, mask, color="#ff000000")
    plt.xlabel(r"$\tau$", fontsize=20)
    plt.ylabel("Objective function", fontsize=18)
    plt.tick_params(labelsize=12)
    plt.ylim(0,)
    plt.subplots_adjust(left=0.15, bottom=0.12)

    # save or show
    if args.out_path is not None:
        fig.savefig(args.out_path, dpi=300)
    else:
        plt.show()
