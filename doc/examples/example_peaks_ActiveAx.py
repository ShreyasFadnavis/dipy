# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:00:37 2017

@author: mafzalid
"""
import numpy as np
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask
from dipy.reconst.dti import TensorModel
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.core.geometry import cart2sphere
from dipy.direction import peaks_from_model
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.segment.mask import median_otsu
from dipy.viz import actor, window
from dipy.io.image import save_nifti
from nibabel.streamlines import save as save_trk
from dipy.core.ndindex import ndindex

#from nibabel.streamlines import Tractogram
#from dipy.tracking.streamline import Streamlines
from time import time
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
# import dipy.reconst.activeax as activeax
import dipy.reconst.activeax_crossing_in_vivo_3compartments as \
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

maskdata, mask = median_otsu(data, 3, 1, False,
                             vol_idx=range(10, 50), dilate=2)
#Y = data[:,:,31,:]
interactive = False
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


theta = np.zeros((data.shape[0], data.shape[1], 1, 5))
phi = np.zeros((data.shape[0], data.shape[1], 1, 5))
num_peaks = np.zeros((data.shape[0], data.shape[1], 1))

for i, j, k in ndindex((data.shape[0], data.shape[1], 1)):
    if mask[i, j, 31] > 0:
        n = 0
        for m in range(5):
            x = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 0])
            y = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 1])
            z = np.squeeze(csd_peaks.peak_dirs[i, j, k, m, 2])
            if (x**2 + y**2 + z**2) > 0:
                r, theta[i, j, k, m], phi[i, j, k, m] = cart2sphere(x, y, z)
                n = n + 1
                num_peaks[i, j, k] = n
#            theta[i, j, k, m] = np.arctan(Y / X)
#            phi[i, j, k, m] = np.arctan(np.sqrt(X**2 + Y**2)/Z)
