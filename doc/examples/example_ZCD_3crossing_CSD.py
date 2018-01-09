# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:46:00 2018

@author: mafzalid
"""

from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
from dipy.segment.mask import median_otsu
import dipy.reconst.ZCD_3crossing_CSD as ZCD_3crossing_CSD
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.core.geometry import cart2sphere
from dipy.direction import peaks_from_model
from dipy.data import get_sphere
from dipy.core.ndindex import ndindex
# t1 = time()

D_intra = 1.7 * 10 ** 3
D_iso = 3 * 10 ** 3

fname, fscanner = get_data('ZCDx_synth_data')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data = img.get_data()

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
#    sigma = np.std(b_zero_all1)
#    y[C] = 1
    return y

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

sphere = get_sphere('repulsion724')
maskdata, mask = median_otsu(data, 3, 1, False, vol_idx=range(10, 50),
                             dilate=2)
mask = data[:, :, :, 0]
csd_peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=sphere,
                             mask=mask,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)

theta_angle = np.zeros((data.shape[0], data.shape[1], 1, 5))
phi_angle = np.zeros((data.shape[0], data.shape[1], 1, 5))
num_peaks = np.zeros((data.shape[0], data.shape[1], 1))

for i, j, k in ndindex((data.shape[0], data.shape[1], 1)):
    if mask[i, j, 0] > 0:
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

fit_method = 'MIX'

Y = data
activeax_fit = np.zeros((Y.shape[0], Y.shape[1], 1, 16))

t1 = time()
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        for k in range(1):
            if mask[i, j, 0] > 0:
                signal = np.squeeze(np.array(Y[i, j]))
                signal = norm_meas_HCP(signal, bvals)
                signal = np.float64(signal)
                signal[signal > 1] = 1
                activeax_model = ZCD_3crossing_CSD.ActiveAxModel(gtab, params,
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
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 0], affine), 'f11_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 1], affine), 'f12_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 2], affine), 'theta1_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 3], affine), 'phi1_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 4], affine), 'R1_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 5], affine), 'f21_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 6], affine), 'f22_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 7], affine), 'f3_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 8], affine), 'theta2_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 9], affine), 'phi2_ZCD_3crossing_CSD.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 10], affine), 'R2_ZCD_3crossing_CSD.nii.gz')

nib.save(nib.Nifti1Image(phi_angle, affine), 'ZCDx_synth_phi_angle.nii.gz')
nib.save(nib.Nifti1Image(theta_angle, affine), 'ZCDx_synth_theta_angle.nii.gz')

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
