from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
from dipy.segment.mask import median_otsu
# import dipy.reconst.activeax as activeax
import dipy.reconst.activeax_crossing as \
    activeax_crossing_in_vivo_3compartments
from scipy.linalg import get_blas_funcs
gemm = get_blas_funcs("gemm")
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


Y = data[:, :, 31, :]
maskdata, mask = median_otsu(Y, 3, 1, False,
                             vol_idx=range(10, 50), dilate=2)

# fname = get_data('mask_CC_in_vivo')
# img = nib.load(fname)
# mask = img.get_data()

fit_method = 'MIX'
activeax_model = activeax_crossing_in_vivo_3compartments.ActiveAxModel(gtab, params, D_intra, D_iso, fit_method=fit_method)
activeax_fit = np.zeros((Y.shape[0], Y.shape[1], 1, 11))
DI1 = np.zeros((Y.shape[0], Y.shape[1], 1))
DI2 = np.zeros((Y.shape[0], Y.shape[1], 1))

#mask[0:127, :] = mask[128:0:-1, :]

t1 = time()
for i in range(65, 85):  #(Y.shape[0]):
    for j in range(75, 95):  #(Y.shape[1]):
        for k in range(1):
            if mask[i, j, 31] > 0:
                signal = np.array(Y[i, j])
                signal = np.squeeze(signal)
                signal = norm_meas_HCP(signal, bvals)
                signal = np.float64(signal)
                signal[signal > 1] = 1
#               signal_n = add_noise(signal, snr=20, noise_type='rician')
                activeax_fit[i, j, k, :] = activeax_model.fit(signal)
                result = np.squeeze(activeax_fit[i, j, k, :])
                if result[4] < result[10]:
                    activeax_fit[i, j, k, 0:2] = result[5:7]
                    activeax_fit[i, j, k, 2:5] = result[8:11]
                    activeax_fit[i, j, k, 5:7] = result[:2]
                    activeax_fit[i, j, k, 8:11] = result[2:5]
                DI1[i, j, k] = activeax_fit[i, j, k, 0] / (activeax_fit[i, j, k, 0] + activeax_fit[i, j, k, 1]) / (np.pi*(activeax_fit[i, j, k, 4]**2))
                DI2[i, j, k] = activeax_fit[i, j, k, 5] / (activeax_fit[i, j, k, 5] + activeax_fit[i, j, k, 6]) / (np.pi*(activeax_fit[i, j, k, 10]**2))
                print(i)
t2 = time()
fast_time = t2 - t1
print(fast_time)
#print(activeax_fit[0, 0])

affine = img.affine.copy()
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 0], affine), 'f11_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 1], affine), 'f12_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 2], affine), 'theta1_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 3], affine), 'phi1_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 4], affine), 'R1_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 5], affine), 'f21_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 6], affine), 'f22_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 7], affine), 'f3_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 8], affine), 'theta2_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 9], affine), 'phi2_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 10], affine), 'R2_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(DI1, affine), 'DI1_in_vivo_3comp_activeax_crossing.nii.gz')
nib.save(nib.Nifti1Image(DI2, affine), 'DI2_in_vivo_3comp_activeax_crossing.nii.gz')
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