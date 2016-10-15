# -*- encoding: utf-8 -*-
#
# Copyright (c) 2011, Jiaya Jia, The Chinese University of Hong Kong.
# Copyright (c) 2016, Andrey Alekseenko
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and / or other materials provided with the distribution.
#
#     * Neither the names of the copyright holders nor the names of the
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np


def _psf2otf(psf, dim):
    """Based on the matlab function of the same name"""
    h, w = psf.shape
    mat = np.zeros(dim)
    mat[:h, :w] = psf
    mat = np.roll(mat, -(w//2), axis=1)
    mat = np.roll(mat, -(h//2), axis=0)
    otf = np.fft.fft2(mat)
    return otf


def l0smooth(im, l=0.02, k=2.0):
    """Smooth image using L0 gradient minimization

    Args:
        im: Input uint8 image, grayscale or RGB/BGR. HSV typically does not work.
        l: Smoothing parameter controlling the degree of smooth (lambda)
           Typically it is within the range [0.001, 0.1], 0.02 by default.
        k: Parameter that controls the rate (kappa)
           Small kappa results in more iterations and with sharper edges.
           We select k in (1, 2], 2 is suggested for natural images.
    Returns:
        Smoothed float32 image
    Reference:
        Xu, L., Lu, C., Xu, Y., Jia, J., 2011. Image smoothing via L0 gradient minimization.
            Proc. 2011 SIGGRAPH Asia Conf. - SA ’11 30, 1. doi:10.1145/2024156.2024208
        Code based on OpenCV L0Smooth module (cv::ximgproc::l0Smooth)
    """

    beta_max = 1e5
    im = im.astype('float') / 255.
    im = np.atleast_3d(im)
    channels = im.shape[2]
    kernel_inv = np.array([[1., -1.]])
    otf_fx = _psf2otf(kernel_inv,   im.shape[:2])
    otf_fy = _psf2otf(kernel_inv.T, im.shape[:2])
    denom_const = np.abs(otf_fx)**2 + np.abs(otf_fy)**2
    denom_const = np.atleast_3d(denom_const)
    if channels > 1:
        denom_const = np.tile(denom_const, (1, 1, channels))
    numer_const = np.fft.fft2(im, axes=(0, 1))
    beta = 2*l

    while beta < beta_max:
        denom = 1 + beta*denom_const

        # h-v subproblem
        h = np.roll(im, -1, axis=1) - im
        v = np.roll(im, -1, axis=0) - im
        hv_mag = (h**2+v**2)
        if channels == 1:
            mask = hv_mag < l/beta
        else:
            mask = np.sum(hv_mag, axis=2) < l/beta
            mask = np.tile(np.atleast_3d(mask), (1, 1, channels))
        h[mask] = v[mask] = 0

        # im subproblem
        hv_grad = - (h - np.roll(h, 1, axis=1) + v - np.roll(v, 1, axis=0))
        hv_grad_freq = np.fft.fft2(hv_grad, axes=(0, 1))
        numer = numer_const + beta*hv_grad_freq
        im = np.real(np.fft.ifft2(numer/denom, axes=(0, 1)))
        beta *= k

    return im
