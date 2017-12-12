# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:34:12 2017

@author: mafzalid
"""
from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
from dipy.segment.mask import median_otsu
import dipy.reconst.activeax_3crossing_in_vivo_CSD as \
    activeax_3crossing_in_vivo_CSD
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.core.geometry import cart2sphere
from dipy.direction import peaks_from_model
from dipy.data import get_sphere
from dipy.core.ndindex import ndindex
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
# signal_param = mix.make_signal_param(signal, bvals, bvecs, G, small_delta,
#                                     big_delta)
#am = np.array([1.84118307861360])


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
#    sigma = np.std(b_zero_all1)
#    y[C] = 1
    return y

#Y = data[:, :, 31, :]
#mask = mask[:, :, 31]

response, ratio = auto_response(gtab, data[:, :, 31:32, :], roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

sphere = get_sphere('repulsion724')
maskdata, mask = median_otsu(data, 3, 1, False, vol_idx=range(10, 50),
                             dilate=2)
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



#fname = get_data('mask_CC_in_vivo')
#img = nib.load(fname)
#mask = img.get_data()
fit_method = 'MIX'
#activeax_model = activeax_crossing_in_vivo_3compartments_CSD.ActiveAxModel(gtab, params,
#                                                                  D_intra,
#                                                                  D_iso,
#                                                                  theta_angle[i, j, k],
#                                                                  phi_angle[i, j, k],
#                                                                  num_peaks[i, j, k],
#                                                                  fit_method=fit_method)
#activeax_model = activeax_crossing_in_vivo.ActiveAxModel(gtab, params, D_intra, D_iso, fit_method=fit_method)
Y = data[:, :, 31:32, :]
activeax_fit = np.zeros((Y.shape[0], Y.shape[1], 1, 16))
#activeax_model = activeax_crossing_in_vivo_3compartments.ActiveAxModel(gtab, params, D_intra, D_iso, fit_method=fit_method)
#mask[0:127, :] = mask[128:0:-1, :]

i = 65
j = 83
k = 0

t1 = time()
for i in range(65, 85):  #(Y.shape[0]):
    for j in range(75, 95):  #(Y.shape[1]):
        for k in range(1):
            if mask[i, j, 31] > 0:
                signal = np.squeeze(np.array(Y[i, j]))
                signal = norm_meas_HCP(signal, bvals)
                signal = np.float64(signal)
                signal[signal > 1] = 1
#               signal_n = add_noise(signal, snr=20, noise_type='rician')
                activeax_model = activeax_3crossing_in_vivo_CSD.ActiveAxModel(gtab, params,
                                                                  D_intra,
                                                                  D_iso,
                                                                  theta_angle[i, j, k],
                                                                  phi_angle[i, j, k],
                                                                  num_peaks[i, j, k],
                                                                  fit_method=fit_method)
                activeax_fit[i, j, k, :] = activeax_model.fit(signal)
                result = np.squeeze(activeax_fit[i, j, k, :])
                if result[4] < result[10]:
                    activeax_fit[i, j, k, 0:2] = result[5:7]
                    activeax_fit[i, j, k, 2:5] = result[8:11]
                    activeax_fit[i, j, k, 5:7] = result[:2]
                    activeax_fit[i, j, k, 8:11] = result[2:5]
                print(i)
t2 = time()
fast_time = t2 - t1
print(fast_time)
#print(activeax_fit[0, 0])



affine = img.affine.copy()
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 0], affine), 'f11_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 1], affine), 'f12_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 2], affine), 'theta1_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 3], affine), 'phi1_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 4], affine), 'R1_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 5], affine), 'f21_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 6], affine), 'f22_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 7], affine), 'f3_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 8], affine), 'theta2_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 9], affine), 'phi2_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 10], affine), 'R2_in_vivo_3comp_activeax_crossing_CSD_ROI.nii.gz')

nib.save(nib.Nifti1Image(phi_angle[:, :, :, :], affine), 'activeax_phi_angle.nii.gz')
nib.save(nib.Nifti1Image(theta_angle[:, :, :, :], affine), 'activeax_theta_angle.nii.gz')

#t2 = time()
#fast_time = t2 - t1
#print(fast_time)
#plt.imshow(data[:, :, 0], cmap='autumn', vmin=0, vmax=8); colorbar()

#import matplotlib.pyplot as plt
#plt.plot(signal)
#plt.ylabel('some numbers')
#plt.show()

#%matplotlib inline
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#imgplot = plt.imshow(Y[:,:,0])
#imgplot = plt.imshow(mask[:,:,0])
