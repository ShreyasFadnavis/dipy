# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:09:07 2017

@author: mafzalid
"""

import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
from scipy.optimize import differential_evolution
import dipy.reconst.activeax_crossing_in_vivo_3compartments as \
    activeax_crossing_in_vivo_3compartments
#import dipy.reconst.activeax_crossing_in_vivo as activeax_crossing_in_vivo
#import dipy.reconst.activeax_in_vivo_3compartments as activeax_in_vivo_3compartments
#from scipy.linalg import get_blas_funcs
from scipy.optimize import least_squares
#from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
#                                   auto_response)
#from dipy.core.geometry import cart2sphere
#from dipy.direction import peaks_from_model
#from dipy.data import get_sphere
#from dipy.core.ndindex import ndindex

#gemm = get_blas_funcs("gemm")
from dipy.segment.mask import median_otsu
# t1 = time()

D_intra = 1.7 * 10 ** 3
D_iso = 3 * 10 ** 3

fname, fscanner = get_data('ActiveAx_in_vivo1')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data1 = img.get_data()

fname = get_data('ActiveAx_in_vivo2')
img = nib.load(fname)
data2 = img.get_data()

fname = get_data('ActiveAx_in_vivo3')
img = nib.load(fname)
data3 = img.get_data()

fname = get_data('ActiveAx_in_vivo4')
img = nib.load(fname)
data4 = img.get_data()

data = np.zeros((data1.shape[0], data1.shape[1], data1.shape[2], data1.shape[3]*4))

data[:, :, :, 0: data1.shape[3]] = data1
data[:, :, :, data1.shape[3]: data1.shape[3]*2] = data2
data[:, :, :, 2*data1.shape[3]: data1.shape[3]*3] = data3
data[:, :, :, 3*data1.shape[3]: data1.shape[3]*4] = data4

affine = img.affine
bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6  # gradient strength
big_delta = params[:, 4]
small_delta = params[:, 5]
te = params[:, 6]
gamma = 2.675987 * 10 ** 8
bvals = gamma ** 2 * G ** 2 * small_delta ** 2 * (big_delta - small_delta / 3.)
bvals = bvals
gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta,
                      b0_threshold=0, atol=1e-2)

#Y = data[63, :, :, :]


def norm_meas_HCP(ydatam, b):

    """
    calculates std of the b0 measurements and normalizes all ydatam
    """
    b1 = np.where(b > 1e-5)
    b2 = range(b.shape[0])
    C = np.setdiff1d(b2, b1)
    b_zero_all = ydatam[C]
    b_zero_norm = sum(b_zero_all) / C.shape[0]
    y = ydatam / b_zero_norm
    return y

#response, ratio = auto_response(gtab, data[:, :, 31:32, :], roi_radius=10, fa_thr=0.7)
#csd_model = ConstrainedSphericalDeconvModel(gtab, response)
#
#sphere = get_sphere('repulsion724')
#maskdata, mask = median_otsu(data, 3, 1, False,
#                             vol_idx=range(10, 50), dilate=2)
#csd_peaks = peaks_from_model(model=csd_model,
#                             data=data[:, :, 31:32, :],
#                             sphere=sphere,
#                             mask=mask[:, :, 31:32],
#                             relative_peak_threshold=.5,
#                             min_separation_angle=25,
#                             parallel=True)
#
#
#theta_angle = np.zeros((data.shape[0], data.shape[1], 1, 5))
#phi_angle = np.zeros((data.shape[0], data.shape[1], 1, 5))
#num_peaks = np.zeros((data.shape[0], data.shape[1], 1))
#
#for i, j, k in ndindex((data.shape[0], data.shape[1], 1)):
#    if mask[i, j, 31] > 0:
#        n = 0
#        for m in range(5):
#            x = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 0])
#            y = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 1])
#            z = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 2])
#            if (x**2 + y**2 + z**2) > 0:
#                r, theta_angle[i, j, k, m], phi_angle[i, j, k, m] = cart2sphere(x, y, z)
#                phi_angle[i, j, k, m] = phi_angle[i, j, k, m] + np.pi
#                if phi_angle[i, j, k, m] > np.pi:
#                    phi_angle[i, j, k, m] = phi_angle[i, j, k, m] - np.pi
#                n = n + 1
#                num_peaks[i, j, k] = n

#i = 60
#j = 80
#k = 0

i = 53
j = 33
k = 0

fit_method = 'MIX'
activeax_model = activeax_crossing_in_vivo_3compartments.ActiveAxModel(gtab, params,
                                                              D_intra, D_iso,                                                                
                                                fit_method=fit_method)
#activeax_model = activeax_crossing_in_vivo.ActiveAxModel(gtab, params, D_intra, D_iso, fit_method=fit_method)
Y = data[:, :, 31:32, :]
activeax_fit = np.zeros((Y.shape[0], Y.shape[1], 1, 6))
#activeax_fit = np.zeros((Y.shape[0], Y.shape[1], 1, 12))


signal = np.squeeze(np.array(Y[i, j]))
signal = norm_meas_HCP(signal, bvals)
signal = np.float64(signal)
signal[signal > 1] = 1
#activeax_fit[i, j, k, :] = activeax_model.fit(signal)

#bounds = np.array([(0.011, np.pi), (0.011, np.pi), (0.11, 16),
#                          (0.11, 0.8)])
bounds = np.array([(0.011, np.pi), (0.011, np.pi), (0.11, 15.9),
                          (0.11, 0.8), (0.011, np.pi), (0.011, np.pi),
                          (0.11, 15.9), (0.11, 0.8)])
res_one = differential_evolution(activeax_model.stoc_search_cost, bounds,
                                 maxiter=activeax_model.maxiter,
                                 args=(signal,))
x = res_one.x

#x = np.array([5, 0.5])
phi = activeax_model.Phi(x)
fe = activeax_model.cvx_fit(signal, phi)

x_fe1 = activeax_model.x_and_fe_to_x_fe(x, fe)
#bounds = ([0.01,  0.01, 0.01, 0.01, 0.01, 0.1],
#          [0.9, 0.9, 0.9, np.pi, np.pi, 16])
bounds = ([0.01, 0.01,  0.01, 0.01, 0.1, 0.01,  0.01, 0.01, 0.01,
                       0.01, 0.1], [0.9,  0.9, np.pi, np.pi, 16, 0.9, 0.9, 0.9,
                                    np.pi, np.pi, 16])
res = least_squares(activeax_model.nlls_cost, x_fe1, bounds=(bounds),
                    xtol=activeax_model.xtol, args=(signal,))
result1 = res.x

#%matplotlib inline
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#imgplot = plt.imshow(Y[:,:,0])
#imgplot = plt.imshow(mask[:,:,0])
