# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:57:58 2017

@author: mafzalid
"""
# -*- coding: utf-8 -*-
from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
# import dipy.reconst.activeax as activeax
import dipy.reconst.activeax_fast as activeax_fast
from scipy.linalg import get_blas_funcs
gemm = get_blas_funcs("gemm")
from dipy.segment.mask import median_otsu
# t1 = time()

#fname = get_data('mask_CC')
#img = nib.load(fname)
#mask = img.get_data()

fname, fscanner = get_data('ActiveAx_monkey_b13183')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data1 = img.get_data()

fname, fscanner = get_data('ActiveAx_monkey_b1925')
img = nib.load(fname)
data2 = img.get_data()

fname, fscanner = get_data('ActiveAx_monkey_b1931')
img = nib.load(fname)
data3 = img.get_data()

fname, fscanner = get_data('ActiveAx_monkey_b3091')
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


def norm_meas_Aax(signal):

    """
    normalizing the signal based on the b0 values of each shell
    """
    y = signal
    y01 = (y[0] + y[1] + y[2]) / 3
    y02 = (y[93] + y[94] + y[95]) / 3
    y03 = (y[186] + y[187] + y[188]) / 3
    y04 = (y[279] + y[280] + y[281]) / 3
    y1 = y[0:93] / y01
    y2 = y[93:186] / y02
    y3 = y[186:279] / y03
    y4 = y[279:372] / y04
    f = np.concatenate((y1, y2, y3, y4))
    return f

#maskdata, mask = median_otsu(data, 3, 1, False,
#                             vol_idx=range(10, 50), dilate=2)
fname = get_data('mask_CC_monkey')
img = nib.load(fname)
mask = img.get_data()
    
fit_method = 'MIX'
activeax_model = activeax_fast.ActiveAxModel(gtab, params, fit_method=fit_method)
#activeax_model = activeax.ActiveAxModel(gtab, fit_method=fit_method)
activeax_fit = np.zeros((data.shape[0], data.shape[1], data.shape[2], 7))

t1 = time()
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            if mask[i, j, k] > 0:
                signal = np.array(data[i, j, k])
                signal = norm_meas_Aax(signal)
                signal = np.float64(signal)
#               signal_n = add_noise(signal, snr=20, noise_type='rician')
                activeax_fit[i, j, k, :] = activeax_model.fit(signal)
                print(i)
t2 = time()
fast_time = t2 - t1
print(fast_time)
#print(activeax_fit[0, 0])
affine = img.affine.copy()
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 0], affine), 'f1_Monkey_activeax_CC_slice2.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 1], affine), 'f2_Monkey_activeax_CC_slice2.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 2], affine), 'f3_Monkey_activeax_CC_slice2.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 3], affine), 'theta_Monkey_activeax_CC_slice2.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 4], affine), 'phi_Monkey_activeax_CC_slice2.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 5], affine), 'R_Monkey_activeax_CC_slice2.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 6], affine), 'f4_Monkey_activeax_CC_slice2.nii.gz')
#t2 = time()
#fast_time = t2 - t1
#print(fast_time)
#plt.imshow(data[:, :, 0], cmap='autumn', vmin=0, vmax=8); colorbar()
