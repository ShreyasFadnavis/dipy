from __future__ import division, print_function, absolute_import

import os
import numpy as np
import logging
from dipy.io.image import load_nifti
from dipy.workflows.workflow import Workflow


class IoInfoFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'io_info'

    def run(self, input_files,
            b0_threshold=50, bvecs_tol=0.01, bshell_thr=100):

        """ Provides useful information about different files used in
        medical imaging. Any number of input files can be provided. The
        program identifies the type of file by its extension.

        Parameters
        ----------
        input_files : variable string
            Any number of Nifti1, bvals or bvecs files.
        b0_threshold : float, optional
            (default 50)
        bvecs_tol : float, optional
            Threshold used to check that norm(bvec) = 1 +/- bvecs_tol
            b-vectors are unit vectors (default 0.01)
        bshell_thr : float, optional
            Threshold for distinguishing b-values in different shells
            (default 100)
        """

        np.set_printoptions(3, suppress=True)

        io_it = self.get_io_iterator()

        for input_path in io_it:
            logging.info('------------------------------------------')
            logging.info('Looking at {0}'.format(input_path))
            logging.info('------------------------------------------')

            ipath_lower = input_path.lower()

            if ipath_lower.endswith('.nii') or ipath_lower.endswith('.nii.gz'):

                data, affine, img, vox_sz, affcodes = load_nifti(
                    input_path,
                    return_img=True,
                    return_voxsize=True,
                    return_coords=True)
                logging.info('Data size {0}'.format(data.shape))
                logging.info('Data type {0}'.format(data.dtype))
                logging.info('Data min {0} max {1} avg {2}'
                             .format(data.min(), data.max(), data.mean()))
                logging.info('2nd percentile {0} 98th percentile {1}'
                             .format(np.percentile(data, 2),
                                     np.percentile(data, 98)))
                logging.info('Native coordinate system {0}'
                             .format(''.join(affcodes)))
                logging.info('Affine to RAS1mm \n{0}'.format(affine))
                logging.info('Voxel size {0}'.format(np.array(vox_sz)))
                if np.sum(np.abs(np.diff(vox_sz))) > 0.1:
                    msg = \
                        'Voxel size is not isotropic. Please reslice.\n'
                    logging.warning(msg)

            if os.path.basename(input_path).lower().find('bval') > -1:
                bvals = np.loadtxt(input_path)
                logging.info('Bvalues \n{0}'.format(bvals))
                logging.info('Total number of bvalues {}'.format(len(bvals)))
                shells = np.sum(np.diff(np.sort(bvals)) > bshell_thr)
                logging.info('Number of gradient shells {0}'.format(shells))
                logging.info('Number of b0s {0} (b0_thr {1})\n'
                             .format(np.sum(bvals <= b0_threshold),
                                     b0_threshold))

            if os.path.basename(input_path).lower().find('bvec') > -1:

                bvecs = np.loadtxt(input_path)
                logging.info('Bvectors shape on disk is {0}'
                             .format(bvecs.shape))
                rows, cols = bvecs.shape
                if rows < cols:
                    bvecs = bvecs.T
                logging.info('Bvectors are \n{0}'.format(bvecs))
                norms = np.array([np.linalg.norm(bvec) for bvec in bvecs])
                res = np.where(
                        (norms <= 1 + bvecs_tol) & (norms >= 1 - bvecs_tol))
                ncl1 = np.sum(norms < 1 - bvecs_tol)
                logging.info('Total number of unit bvectors {0}'
                             .format(len(res[0])))
                logging.info('Total number of non-unit bvectors {0}\n'
                             .format(ncl1))

        np.set_printoptions()


class FetchFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'fetch_data'

    def run(self, input_files='fetch_all'):
        """ This workflow is specifically written to ease writing and enabling
        examples in the workflow documentation and examples

        Parameters
        ----------
        input_files : variable string
            Any number of data fetchers already in DIPY
        """

        if input_files == 'fetch_all':
            from dipy.data import (fetch_bundle_atlas_hcp842,
                                   fetch_bundle_fa_hcp,
                                   fetch_bundles_2_subjects,
                                   fetch_cenir_multib, fetch_cfin_multib,
                                   fetch_isbi2013_2shell, fetch_ivim,
                                   fetch_mni_template, fetch_scil_b0,
                                   fetch_sherbrooke_3shell,
                                   fetch_stanford_hardi, fetch_stanford_labels,
                                   fetch_stanford_pve_maps,
                                   fetch_stanford_t1, fetch_taiwan_ntu_dsi,
                                   fetch_syn_data, fetch_target_tractogram_hcp,
                                   fetch_tissue_data)

            # fetching all the data
            fetch_bundle_atlas_hcp842()
            fetch_bundle_fa_hcp()
            fetch_bundles_2_subjects()
            fetch_cenir_multib()
            fetch_cfin_multib(),
            fetch_isbi2013_2shell()
            fetch_isbi2013_2shell()
            fetch_ivim()
            fetch_mni_template()
            fetch_scil_b0()
            fetch_sherbrooke_3shell()
            fetch_stanford_hardi()
            fetch_stanford_labels()
            fetch_stanford_pve_maps()
            fetch_stanford_t1()
            fetch_syn_data()
            fetch_taiwan_ntu_dsi()
            fetch_target_tractogram_hcp()
            fetch_tissue_data()
