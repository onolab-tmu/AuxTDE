# -*- coding: utf-8 -*-
import argparse
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D

import maux_tde
from functions.STFT import mSTFT


def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--init_tau",
        help="Initial time delay estimates",
        type=str,
        default="1.5 3",
    )
    parser.add_argument(
        "-m", "--n_mesh", help="# of mesh for plot", type=int, default=50,
    )
    parser.add_argument(
        "-d",
        "--divides_obj",
        help="With this argument, graphs are divided corresponding to the vertex of the auxiliary function",
        action="store_true",
    )
    parser.add_argument(
        "-o", "--out_path", type=str, help="Path for save figure", default=None,
    )
    parser.add_argument(
        "-a", "--azim", help="Azimuth of 3D plot", type=int, default=-30,
    )
    parser.add_argument(
        "-e", "--elev", help="Elevation of 3D plot", type=int, default=25,
    )
    parser.add_argument(
        "-w",
        "--n_worker",
        help="# of workers for parallel processing",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-z", "--z_label_pos", type=float, default=0.7,
    )
    return parser.parse_args()


def cost_function(args):
    x = args[0]
    y = args[1]
    tau = np.array([0, x, y])

    return maux_tde.cost_function(a, tau, A, phi, w)


def auxiliary_function(args):
    x = args[0]
    y = args[1]
    tau = np.array([0, x, y])

    return maux_tde.auxiliary_function(a, tau, init_tau, A, phi, w)


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
    n_ch = 3
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
    V = maux_tde.calc_SCM(X)
    A = np.abs(V)
    phi = np.angle(V / A)
    A /= frlen
    A[:, :, 1:-1] *= 2
    a = np.ones([n_freq, n_ch, 1])

    ## Objective function
    # set range
    range_1 = np.linspace(-3, 5, args.n_mesh)
    range_2 = np.linspace(-3, 5, args.n_mesh)
    n_row = len(range_2)
    n_col = len(range_1)
    X, Y = np.meshgrid(range_1, range_2)

    # compute objective function
    cost = np.zeros([n_row, n_col])
    with Pool(args.n_worker, initializer=mp_init) as p:
        cost = p.map(cost_function, list(zip(np.ravel(X), np.ravel(Y))))
    cost = np.array(cost).reshape(n_col, n_row)

    ## auxiliary function
    taus = args.init_tau.split()
    init_tau = [0]
    for i in taus:
        init_tau = np.append(init_tau, float(i))
    init_tau = np.array(init_tau)

    # compute auxiliary function
    af = np.zeros([n_row, n_col])
    with Pool(args.n_worker, initializer=mp_init) as p:
        af = p.map(auxiliary_function, list(zip(np.ravel(X), np.ravel(Y))))
    af = np.array(af).reshape(n_col, n_row)
    af *= af > 0

    ## plot
    # set colors
    colors_1 = np.full(X.shape, "#0000ff99")
    colors_2 = np.full(X.shape, "#eb3929ff")
    if args.divides_obj:
        colors_1[X > init_tau[1]] = "#0000ff00"
        colors_2[X > init_tau[1]] = "#0000ff00"
    colors_2[af <= 0] = "#ff000000"

    # plot
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = Axes3D(fig)

    ax.plot_surface(
        X,
        Y,
        cost,
        facecolors=colors_1,
        rcount=args.n_mesh,
        ccount=args.n_mesh,
        linewidths=0,
    )
    ax.plot_surface(
        X, Y, af, facecolors=colors_2, rcount=args.n_mesh, ccount=args.n_mesh
    )

    ax.set_xlabel(r"$\tau_2$", fontsize=20)
    ax.set_ylabel(r"$\tau_1$", fontsize=20)
    ax.set_zlabel("Objective function", fontsize=18, labelpad=-3)
    ax.tick_params(labelsize=12, pad=-2)

    ax.set_xlim3d([-3, 5])
    ax.set_zlim3d([0, np.amax(cost)])

    # settings for sci notation
    ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="z", scilimits=(3, 3))
    ax.zaxis.offsetText.set_visible(False)
    ax.text2D(
        0.97, args.z_label_pos, r"$\times 10^{3}$", transform=ax.transAxes, fontsize=12
    )

    ax.view_init(azim=args.azim, elev=args.elev)

    # save or show
    if args.out_path is not None:
        fig.savefig(args.out_path, dpi=300)
    else:
        plt.show()
