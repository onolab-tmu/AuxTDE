# -*- coding: utf-8 -*-
"STFT: Package for short-time Fourier Transform"
import numpy as np
import matplotlib.pyplot as plt


def STFT(sig, frlen, frsft, wnd, zp=True):
    """
    short-time Fourier Transform

    parameters
    ----------
    sig: array_like (n_samples)
        Time domain signal to be analyzed
    frlen: int
        Frame length of STFT analysis (samples)
    frsft: int
        Frame shift length of STFT analysis (samples)
    wnd: array_like (n_samples)
        Window function, the length must be equal to 'frlen'
    zp: bool, optional
        If True, do zero padding at the head of 'sig'

    return
    ----------
    SIG: array_like (# of frames, # of freq. bin)
        STFT domain signal
    """

    # zero padding
    l_zp = frlen - frsft if zp is True else 0
    zero_pad = np.zeros(l_zp)
    sig = np.concatenate([zero_pad, sig, zero_pad])
    n_samples = sig.shape[0]

    # deal with an abnormal case
    if frlen > n_samples:
        sig = np.append(sig, np.zeros([frlen - n_samples]))
        n_samples = frlen

    # set the number of time frames and frequency bins
    n_frame = int(np.ceil((n_samples - frsft) / frsft))
    n_freq = frlen // 2 + 1

    # zero padding for the tail of the signal
    tmp = np.zeros([(n_frame - 1) * frsft + frlen - n_samples])
    sig = np.concatenate([sig, tmp])

    # initialization
    SIG = np.zeros([n_frame, n_freq], dtype=complex)

    # main
    for t in range(n_frame):
        head = t * frsft
        SIG[t, :] = np.fft.rfft(sig[head : head + frlen] * wnd)

    return SIG


def mSTFT(msig, frlen, frsft, wnd, zp=True):
    """
    short-time Fourier Transform

    parameters
    ----------
    msig: array_like (n_ch, n_samples)
        Time domain multichannel signal to be analyzed
    frlen: int
        Frame length of STFT analysis (samples)
    frsft: int
        Frame shift length of STFT analysis (samples)
    wnd: array_like (n_samples)
        Window function, the length must be equal to 'frlen'
    zp: bool, optional
        If True, do zero padding at the head of 'msig'

    return
    ----------
    mSIG: array_like (# of ch., # of frames, # of freq. bin)
        STFT domain signal
    """

    # set parameters
    if msig.ndim == 1:
        n_samples = msig.shape[0]
        n_ch = 1
        msig = msig[np.newaxis, :]
    else:
        n_ch, n_samples = msig.shape
        if n_samples < n_ch:
            msig = msig.T
            n_ch, n_samples = msig.shape

    l_zp = frlen - frsft if zp is True else 0

    # deal with an abnormal case
    if frlen > n_samples:
        msig = np.append(msig, np.zeros([n_ch, frlen - n_samples]), axis=0)
        n_samples = frlen

    # set the number of time frames and frequency bins
    n_frame = int(np.ceil((n_samples + 2 * l_zp - frsft) / frsft))
    n_freq = frlen // 2 + 1

    # initialization
    mSIG = np.zeros([n_ch, n_frame, n_freq], dtype=complex)

    # main
    for ch in range(n_ch):
        mSIG[ch, :, :] = STFT(msig[ch, :], frlen, frsft, wnd, zp)

    return mSIG if n_ch != 1 else np.squeeze(mSIG, axis=0)


def iSTFT(SIG, frsft, wnd, zp=True):
    """
    inverse short-time Fourier Transform

    parameters
    ----------
    SIG: array_like (# of frames, # of freq. bin)
        STFT domain signal
    frsft: int
        Frame shift length used for STFT analysis (samples)
    wnd: array_like (n_samples)
        Window function satisfying perfect re-construction property
    zp: bool, optional
        If zero padding was performed in STFT(), then set True

    return
    ----------
    sig: array_like (n_samples)
        Time domain signal
    """

    # set default values
    n_frame, n_freq = SIG.shape
    frlen = (n_freq - 1) * 2
    l_zp = frlen - frsft if zp is True else 0

    # initialization
    sig = np.zeros([(n_frame - 1) * frsft + frlen])
    wnd = sync_wnd(wnd, frsft)

    # main
    for t in range(n_frame):
        head = t * frsft
        sig[head : head + frlen] += np.fft.irfft(SIG[t, :], frlen) * wnd

    return sig[l_zp:-l_zp] if zp else sig


def miSTFT(mSIG, frsft, wnd, zp=True):
    """
    inverse short-time Fourier Transform for multichannel signal

    parameters
    ----------
    mSIG: array_like (# of ch., # of frames, # of freq. bin)
        STFT domain multichannel signal
    frsft: int
        Frame shift length used for STFT analysis (samples)
    wnd: array_like (n_samples)
        Window function satisfying perfect re-construction property
    zp: bool, optional
        If zero padding was performed in STFT(), then set True

    return
    ----------
    msig: array_like (n_samples)
        Time domain multichannel signal
    """

    # set default values
    if mSIG.ndim == 2:
        n_frame, n_freq = mSIG.shape
        n_ch = 1
        mSIG = mSIG[np.newaxis, :, :]
    else:
        n_ch, n_frame, n_freq = mSIG.shape
    frlen = (n_freq - 1) * 2
    l_zp = frlen - frsft if zp is True else 0

    # initialization
    msig = np.zeros([n_ch, (n_frame - 1) * frsft + frlen - 2 * l_zp])

    # main
    for k in range(n_ch):
        msig[k, :] = iSTFT(mSIG[k, :, :], frsft, wnd, zp)

    return np.squeeze(msig)


def sync_wnd(wnd, frsft):
    frlen = wnd.shape[0]
    sync_wnd = np.zeros(frlen)

    for i in range(frsft):
        amp = 0
        for j in range(frlen // frsft):
            amp = amp + wnd[i + (j - 1) * frsft] * wnd[i + (j - 1) * frsft]
        for j in range(frlen // frsft):
            sync_wnd[i + (j - 1) * frsft] = wnd[i + (j - 1) * frsft] / amp

    return sync_wnd


def truncate(sig, ref):
    return sig[: ref.shape[0]] if sig.ndim == 1 else sig[:, : ref.shape[0]]
