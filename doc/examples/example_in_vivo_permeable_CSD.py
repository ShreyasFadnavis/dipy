# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:08:05 2017

@author: mafzalid
"""
from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
import dipy.reconst.activeax_in_vivo_permeable_CSD as \
activeax_in_vivo_permeable_CSD
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.core.geometry import cart2sphere
from dipy.direction import peaks_from_model
from dipy.data import get_sphere
from dipy.core.ndindex import ndindex
# t1 = time()

#D_intra = 1.7 * 10 ** 3
D_intra = 2 * 10 ** 3
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
gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta,
                      b0_threshold=0, atol=1e-2)


def norm_meas_HCP(ydatam, b):

    """
    calculates std of the b0 measurements and normalizes all ydatam
    """
    b1 = np.where(b > 1e-5)
    b2 = range(b.shape[0])
    C = np.setdiff1d(b2, b1)
    b_zero_all = ydatam[C]
#    %sigma = std(b_zero_all, 1)
    b_zero_norm = sum(b_zero_all) / C.shape[0]
    y = ydatam / b_zero_norm
#    b_zero_all1 = y[C]
#    y[C] = 1
#    sigma = np.std(b_zero_all1)
    return y


maskdata, mask = median_otsu(data, 3, 1, False,
                             vol_idx=range(10, 50), dilate=2)
#Y = data[:, :, 31, :]
#mask = mask[:, :, 31]

response, ratio = auto_response(gtab, data[:, :, 31:32, :], roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

sphere = get_sphere('repulsion724')

csd_peaks = peaks_from_model(model=csd_model,
                             data=data[:, :, 31:32, :],
                             sphere=sphere,
                             mask=mask[:, :, 31:32],
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)


theta_angle = np.zeros((data.shape[0], data.shape[1], 1, 5))
phi_angle = np.zeros((data.shape[0], data.shape[1], 1, 5))
num_peaks = np.zeros((data.shape[0], data.shape[1], 1))

for i, j, k in ndindex((data.shape[0], data.shape[1], 1)):
    if mask[i, j, 31] > 0:
        n = 0
        for m in range(5):
            x = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 0])
            y = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 1])
            z = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 2])
            if (x**2 + y**2 + z**2) > 0:
                r, theta_angle[i, j, k, m], phi_angle[i, j, k, m] = cart2sphere(x, y, z)
                phi_angle[i, j, k, m] = phi_angle[i, j, k, m] + np.pi
                theta_angle[i, j, k, m] = np.pi - theta_angle[i, j, k, m]
                if phi_angle[i, j, k, m] > np.pi:
                    phi_angle[i, j, k, m] = phi_angle[i, j, k, m] - np.pi
                    theta_angle[i, j, k, m] = np.pi - theta_angle[i, j, k, m]
                n = n + 1
                num_peaks[i, j, k] = n

#maskdata, mask = median_otsu(data, 3, 1, False,
#                             vol_idx=range(10, 50), dilate=2)
#fname = get_data('mask_CC_in_vivo')
#img = nib.load(fname)
#mask = img.get_data()

#Y = data[63,:,:,:]
Y = data[:, :, 31, :]
mask = mask[:, :, 31:32]
fit_method = 'MIX'
#activeax_model = activeax_in_vivo_permeable_CSD.ActiveAxModel(gtab, params, D_intra, D_iso, theta_angle, phi_angle,
#                 num_peaks, fit_method=fit_method)
#activeax_model = activeax.ActiveAxModel(gtab, fit_method=fit_method)
activeax_fit = np.zeros((Y.shape[0], Y.shape[1], 1, 3))
#mask[0:127,:] = mask[128:0:-1,:]

t1 = time()
for i in range(80, 81):  #range(Y.shape[0]):
    for j in range(80, 81):  #range(Y.shape[1]):
        for k in range(1):
            if mask[i, j, k] > 0:
                signal = np.array(Y[i, j])
                signal = norm_meas_HCP(signal, bvals)
                signal = np.float64(signal)
                signal[signal > 1] = 1
                activeax_model = activeax_in_vivo_permeable_CSD.ActiveAxModel(gtab, params,
                                                                  D_intra,
                                                                  D_iso,
                                                                  theta_angle[i, j, k],
                                                                  phi_angle[i, j, k],
                                                                  num_peaks[i, j, k],
                                                                  fit_method=fit_method)
#               signal_n = add_noise(signal, snr=20, noise_type='rician')
                activeax_fit[i, j, k, :] = activeax_model.fit(signal)
                print(i)
t2 = time()
fast_time = t2 - t1
print(fast_time)
#print(activeax_fit[0, 0])
affine = img.affine.copy()
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 0], affine), 'f1_in_vivo_activeax_CC_perme.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 1], affine), 'f2_in_vivo_activeax_CC_perme.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 2], affine), 'f3_in_vivo_activeax_CC_perme.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 3], affine), 'theta_in_vivo_activeax_CC_perme.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 4], affine), 'phi_in_vivo_activeax_CC_perme.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 5], affine), 'R_in_vivo_activeax_CC.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 6], affine), 'f4_in_vivo_activeax_CC.nii.gz')
#t2 = time()
#fast_time = t2 - t1
#print(fast_time)
#plt.imshow(data[:, :, 0], cmap='autumn', vmin=0, vmax=8); colorbar()