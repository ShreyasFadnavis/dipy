from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
import dipy.reconst.activeax as activeax_fast

fname, fscanner = get_data('ActiveAx_synth_2d')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data = img.get_data()
affine = img.affine
bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6  # gradient strength
big_delta = params[:, 4]
small_delta = params[:, 5]
# te = params[:, 6]
gamma = 2.675987 * 10 ** 8
bvals = gamma ** 2 * G ** 2 * small_delta ** 2 * (big_delta - small_delta / 3.)
gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta,
                      b0_threshold=0, atol=1e-2)

D_intra = 0.6 * 10 ** 3
D_iso = 2 * 10 ** 3


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


fit_method = 'MIX'
activeax_model = activeax_fast.ActiveAxModel(gtab, params, D_intra, D_iso,
                                             fit_method=fit_method)
# activeax_model = activeax.ActiveAxModel(gtab, fit_method=fit_method)
activeax_fit = np.zeros((10, 10, 7))
t1 = time()
for i in range(1):
    for j in range(1):
        signal = np.array(data[i, j, 0])
        signal = norm_meas_Aax(signal)
        signal = np.float64(signal)
#        signal_n = add_noise(signal, snr=20, noise_type='rician')
        activeax_fit[i, j, :] = activeax_model.fit(signal)
t2 = time()
fast_time = t2 - t1
print(fast_time)
print(activeax_fit[0, 0])
