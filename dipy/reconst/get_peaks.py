from __future__ import division
import numpy as np
import nibabel as nib
from dipy.data import get_data
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.data import get_sphere
from dipy.sims.voxel import single_tensor_odf
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.viz import window, actor
from dipy.reconst.csdeconv import auto_response

sphere = get_sphere('repulsion724')

fname, fscanner = get_data('small_NODDIx_data')
params = np.loadtxt(fscanner)

# getting the gtab, bvals and bvecs
fbval = '/home/shreyasfadnavis/Desktop/dwi/sub-158035_dwi.bvals'
fbvec = '/home/shreyasfadnavis/Desktop/dwi/sub-158035_dwi.bvecs'

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

for i in range(bvals.shape[0]):
    if bvals[i] > 5:
        bvals[i] = round(bvals[i] / 500.0) * 500.0
    else:
        bvals[i] = bvals[i] - 5
        
bvals = bvals/ 10**6

noddi_data = '/home/shreyasfadnavis/Desktop/dwi/sub-158035_dwi.nii.gz'
img = nib.load(noddi_data)
data = img.get_data()

noddi_mask = \
    '/home/shreyasfadnavis/Desktop/dwi/cc-mask.nii'

img_mask = nib.load(noddi_mask)
mask = img_mask.get_data()

big_delta = params[:, 4]
small_delta = params[:, 5]
gamma = 2.675987 * 10 ** 8
G = params[:, 3] / 10 ** 6

gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta, b0_threshold=0, atol=1e-2)

print("Data Completely Loaded!")

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

# Enables/disables interactive visualization
interactive = True

ren = window.Renderer()
evals = response[0]
evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
response_odf = single_tensor_odf(sphere.vertices, evals, evecs)

data_small = data[20:50, 55:85, 38:39]

csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_peaks = peaks_from_model(model=csd_model,
                             data=data_small,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)

window.clear(ren)
fodf_peaks = actor.peak_slicer(csd_peaks.peak_dirs, csd_peaks.peak_values)
ren.add(fodf_peaks)

def csd_crossings():
    window.record(ren, out_path='csd_peaks.png', size=(600, 600))
    if interactive:
        window.show(ren)

peak_counter = np.linalg.norm(csd_peaks.peak_dirs[..., :, :], axis = -1)
num_peaks = peak_counter.sum(axis = -1)