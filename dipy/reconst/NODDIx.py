<<<<<<< HEAD
from dipy.reconst.base import ReconstModel, ReconstFit 
import numpy as np
import cvxpy as cvx
from dipy.reconst.multi_voxel import multi_voxel_fit
import dipy.reconst.noddi_speed as noddixspeed
=======
from dipy.reconst.base import ReconstModel
import numpy as np
import cvxpy as cvx
import dipy.reconst.noddispeed as noddixspeed
>>>>>>> refs/remotes/origin/nipy-dipy-master
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
from scipy import special

<<<<<<< HEAD
gamma = 2.675987 * 10 ** 8  # gyromagnetic ratio for Hydrogen
D_intra = 1.7 * 10 ** 3  # intrinsic free diffusivity
D_iso = 3 * 10 ** 3  # isotropic diffusivity


# experimental
class NoddixFit(ReconstFit):
    """Diffusion data fit to a NODDIx Model"""

    def __init__(self, model, coeff):
        self.volfrac_ic1 = coeff[0]
        self.volfrac_ec1 = coeff[2]
        self.theta2 = coeff[9]
        self.phi2 = coeff[10]        
        self.volfrac_ic2 = coeff[1]
        self.volfrac_ec2 = coeff[3]
        self.theta1 = coeff[6]  
        self.phi1 = coeff[7]          
        self.volfrac_csf = coeff[4]
        self.OD1 = coeff[5]
        self.OD2 = coeff[8]
        self.coeff = coeff


class NoddixModel(ReconstModel):
=======
gamma = 2.675987 * 10 ** 8
D_intra = 1.7 * 10 ** 3  # (mircometer^2/sec for in vivo human)
D_iso = 3 * 10 ** 3


class NODDIxModel(ReconstModel):
>>>>>>> refs/remotes/origin/nipy-dipy-master
    r""" MIX framework (MIX) [1]_.
    The MIX computes the NODDIx parameters. NODDIx is a multi
    compartment model, (sum of exponentials).
    This algorithm uses three different optimizers. It starts with a
    differential evolution algorithm and fits the parameters in the
    power of exponentials. Then the fitted parameters in the first step are
    utilized to make a linear convex problem. Using a convex optimization,
    the volume fractions are determined. The last step of this algorithm
    is non linear least square fitting on all the parameters.
    The results of the first and second step are utilized as the initial
    values for the last step of the algorithm.
    (see [1]_ for a comparison and a thorough discussion).
<<<<<<< HEAD

    Parameters
    ----------
    ReconstModel of DIPY
    Returns the signal with the following 11 parameters of the model

    Parameters
    ----------
        Volume Fraction 1 - Intracellular 1
        Volume Fraction 2 - Intracellular 2
        Volume Fraction 3 - Extracellular 1
        Volume Fraction 4 - Extracellular 2
        Volume Fraction 5 - CSF: Isotropic
        Orientation Dispersion 1
        Theta 1
        Phi 1
        Orientation Dispersion 2
        Theta 2
        Phi 2

=======
    Parameters
    ----------
    gtab : GradientTable
    fit_method : str or callable
    Returns  the 11 parameters of the model
    -------
>>>>>>> refs/remotes/origin/nipy-dipy-master
    References
    ----------
    .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
           White Matter Fibers from diffusion MRI." Scientific reports 6
           (2016).
<<<<<<< HEAD

    Notes
    -----
    The implementation of NODDIx may require CVXPY (http://www.cvxpy.org/).
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
    """

    def __init__(self, gtab, params, fit_method='MIX'):
        # The maximum number of generations, genetic algorithm 1000 default, 1
<<<<<<< HEAD
        self.maxiter = 100
=======
        self.maxiter = 1000
>>>>>>> refs/remotes/origin/nipy-dipy-master
        # Tolerance for termination, nonlinear least square 1e-8 default, 1e-3
        self.xtol = 1e-8
        self.gtab = gtab
        self.big_delta = gtab.big_delta
        self.small_delta = gtab.small_delta
        self.gamma = gamma
        self.G = params[:, 3] / 10 ** 6  # gradient strength (Tesla/micrometer)
<<<<<<< HEAD
        self.yhat_ball = D_iso * self.gtab.bvals
        self.L = self.gtab.bvals * D_intra
        self.phi_inv = np.zeros((4, 4))
        self.yhat_zeppelin = np.zeros(self.gtab.bvals.shape[0])
        self.yhat_cylinder = np.zeros(self.gtab.bvals.shape[0])
        self.yhat_dot = np.zeros(self.gtab.bvals.shape)
        self.exp_phi1 = np.zeros((self.gtab.bvals.shape[0], 5))
        self.exp_phi1[:, 4] = np.exp(-self.yhat_ball)

    @multi_voxel_fit
    def fit(self, data):
        r""" Fit method of the NODDIx model class

        data : array
        The measured signal from one voxel.

        """
        bounds = [(0.011, 0.98), (0.011, np.pi), (0.011, np.pi), (0.11, 1),
                  (0.011, 0.98), (0.011, np.pi), (0.011, np.pi), (0.11, 1)]

        diff_res = differential_evolution(self.stoc_search_cost, bounds,
                                          maxiter=self.maxiter, args=(data,),
                                          tol=0.001, seed=200,
                                          mutation=(0, 1.05),
                                          strategy='best1bin',
                                          disp=False, polish=True, popsize=14)

        # Step 1: store the results of the differential evolution in x
        x = diff_res.x
        phi = self.Phi(x)
        # Step 2: perform convex optimization
        f = self.cvx_fit(data, phi)
        # Combine all 13 parameters of the model into a single array
=======
        self.G2 = self.G ** 2
        self.yhat_ball = D_iso * self.gtab.bvals
        self.L = self.gtab.bvals * D_intra
        self.phi_inv = np.zeros((4, 4))
        self.yhat_zeppelin = np.zeros(self.small_delta.shape[0])
        self.yhat_cylinder = np.zeros(self.small_delta.shape[0])
        self.yhat_dot = np.zeros(self.gtab.bvals.shape)
        self.exp_phi1 = np.zeros((self.small_delta.shape[0], 5))
        self.exp_phi1[:, 4] = np.exp(-self.yhat_ball)

    def fit(self, data):
        """ Fit method of the NODDIx model class
        Parameters
        ----------
        The 11 parameters that the model outputs after fitting are:
        Volume Fraction 1 - Intracellular 1
        Volume Fraction 2 - Intracellular 2
        Volume Fraction 3 - Extracellular 1
        Volume Fraction 4 - Extracellular 2
        Volume Fraction 5 - CSF: Isotropic
        Orientation Dispersion 1
        Theta 1
        Phi 1
        Orientation Dispersion 2
        Theta 2
        Phi 2
        ----------
        data : array
        The measured signal from one voxel.
        """
        bounds = [(0.011, 0.98), (0.011, np.pi), (0.011, np.pi), (0.11, 1),
                  (0.011, 0.98), (0.011, np.pi), (0.011, np.pi), (0.11, 1)]
        # can we limit this..
        res_one = differential_evolution(self.stoc_search_cost, bounds,
                                         maxiter=self.maxiter, args=(data,))
        x = res_one.x
        phi = self.Phi(x)
        f = self.cvx_fit(data, phi)
>>>>>>> refs/remotes/origin/nipy-dipy-master
        x_f = self.x_and_f_to_x_f(x, f)

        bounds = ([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                   0.01], [0.9, 0.9, 0.9, 0.9, 0.9, 0.99, np.pi, np.pi, 0.99,
                           np.pi, np.pi])
<<<<<<< HEAD
        res = least_squares(self.nlls_cost, x_f, xtol=self.xtol, args=(data,))
        result = res.x
        noddix_fit = NoddixFit(self, result)
        return noddix_fit

    def stoc_search_cost(self, x, signal):
        """
        Cost function for the differential evolution
        Calls another function described by:
            differential_evol_cost
=======
        res = least_squares(self.nlls_cost, x_f, bounds=(bounds),
                            xtol=self.xtol, args=(data,))
        result = res.x
        return result

    def stoc_search_cost(self, x, signal):
        """
        Cost function for the differential evolution | genetic algorithm
        Parameters
        ----------
        x : array
            x.shape = 4x1
            x(0) theta (radian)
            x(1) phi (radian)
            x(2) R (micrometers)
            x(3) v=f1/(f1+f2) (0.1 - 0.8)
        bvals
        bvecs
        G: gradient strength
        small_delta
        big_delta
        gamma: gyromagnetic ratio (2.675987 * 10 ** 8 )
        D_intra= intrinsic free diffusivity (0.6 * 10 ** 3 mircometer^2/sec)
        D_iso= isotropic diffusivity, (2 * 10 ** 3 mircometer^2/sec)
>>>>>>> refs/remotes/origin/nipy-dipy-master
        Returns
        -------
        (signal -  S)^T(signal -  S)
        Notes
        --------
        cost function for differential evolution algorithm:
        .. math::
            (signal -  S)^T(signal -  S)
        """
        phi = self.Phi(x)
        return self.differential_evol_cost(phi, signal)

<<<<<<< HEAD
    def differential_evol_cost(self, phi, signal):

        """
        To make the cost function for differential evolution algorithm
        """
        #  moore-penrose inverse
#        try:
#            phi_mp = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
#        except LinAlgError:
#            from pdb import set_trace
#            set_trace()
#            pass
        phi_mp = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
        #  sigma
        f = np.dot(phi_mp, signal)
        yhat = np.dot(phi, f)
        return np.dot((signal - yhat).T, signal - yhat)

=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
    def cvx_fit(self, signal, phi):
        """
        Linear parameters fit using cvx
        Parameters
        ----------
        phi : array
            phi.shape = number of data points x 4
            signal : array
            signal.shape = number of data points x 1
        Returns
        -------
<<<<<<< HEAD
        f0, f1, f2, f3, f4 (volume fractions)
        f0 = f[0]: Volume Fraction of Intra-Cellular Region 1
        f1 = f[1]: Volume Fraction of Extra-Cellular Region 1
        f2 = f[2]: Volume Fraction of Intra-Cellular Region 2
        f3 = f[3]: Volume Fraction of Extra-Cellular Region 2
        f4 = f[4]: Volume Fraction for region containing CSF
=======
        f1, f2, f3, f4 (volume fractions)
        f1 = f[0]
        f2 = f[1]
        f3 = f[2]
        f4 = f[3]
>>>>>>> refs/remotes/origin/nipy-dipy-master
        Notes
        --------
        cost function for genetic algorithm:
        .. math::
            minimize(norm((signal)- (phi*f)))
        """

        # Create 5 scalar optimization variables.
        f = cvx.Variable(5)
        # Create four constraints.
        constraints = [cvx.sum_entries(f) == 1,
                       f[0] >= 0.011,
                       f[1] >= 0.011,
                       f[2] >= 0.011,
                       f[3] >= 0.011,
                       f[4] >= 0.011,
                       f[0] <= 0.89,
                       f[1] <= 0.89,
                       f[2] <= 0.89,
                       f[3] <= 0.89,
                       f[4] <= 0.89]

        # Form objective.
        obj = cvx.Minimize(cvx.sum_entries(cvx.square(phi * f - signal)))

        # Form and solve problem.
        prob = cvx.Problem(obj, constraints)
        # Returns the optimal value
        prob.solve()
        return np.array(f.value)

    def nlls_cost(self, x_f, signal):
        """
        cost function for the least square problem
        Parameters
        ----------
        x_f : array
<<<<<<< HEAD
            x_f(0) x_f(1) x_f(2) x_f(3) x_f(4) are f1 f2 f3 f4 f5(volfractions)
            x_f(5) Orintation Dispersion 1
            x_f(6) Theta1
            x_f(7) Phi1
            x_f(8) Orintation Dispersion 2
            x_f(9) Theta2
            x_f(10) Phi2
=======
            x_f(0) x_f(1) x_f(2)  are f1 f2 f3
            x_f(3) theta
            x_f(4) phi
            x_f(5) R
            x_f(6) as f4
        signal_param : array
            signal_param.shape = number of data points x 7
            signal_param = np.hstack([signal[:, None], bvals[:, None], bvecs,
                                  G[:, None], small_delta[:, None],
                                  big_delta[:, None]])
>>>>>>> refs/remotes/origin/nipy-dipy-master
        Returns
        -------
        sum{(signal -  phi*f)^2}
        Notes
        --------
        cost function for the least square problem
        .. math::
            sum{(signal -  phi*f)^2}
        """
        x, f = self.x_f_to_x_and_f(x_f)
        phi = self.Phi2(x_f)
        return np.sum((np.dot(phi, f) - signal) ** 2)

<<<<<<< HEAD
    def Phi(self, x):
        """
        Constructs the Signal from the intracellular and extracellular compart-
        ments for the Differential Evolution and Variable Separation.
        """
        self.exp_phi1[:, 0] = self.S_ic1(x)
        self.exp_phi1[:, 1] = self.S_ec1(x)
        self.exp_phi1[:, 2] = self.S_ic2(x)
        self.exp_phi1[:, 3] = self.S_ec2(x)
        return self.exp_phi1

    def Phi2(self, x_f):
        """
        Constructs the Signal from the intracellular and extracellular compart-
        ments: Convex Fitting + NLLS - LM method.
        """
        x, f = self.x_f_to_x_and_f(x_f)
        self.exp_phi1[:, 0] = self.S_ic1(x)
        self.exp_phi1[:, 1] = self.S_ec1_new(x, f)
        self.exp_phi1[:, 2] = self.S_ic2_new(x)
        self.exp_phi1[:, 3] = self.S_ec2_new(x, f)
        return self.exp_phi1

=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
    def S_ic1(self, x):
        """
        This function models the intracellular component.
        The intra-cellular compartment refrs to the space bounded by the
        membrane of neurites. We model this space as a set of sticks, i.e.,
        cylinders of zero radius, to capture the highly restricted nature of
        diffusion perpendicular to neurites and unhindered diffusion along
        them. (see [2]_ for a comparison and a thorough discussion)

        ----------
        References
        ----------
        .. [2] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.

        """
        OD1 = x[0]
        sinT1 = np.sin(x[1])
        cosT1 = np.cos(x[1])
        sinP1 = np.sin(x[2])
        cosP1 = np.cos(x[2])
        n1 = [cosP1*sinT1, sinP1*sinT1, cosT1]
        kappa1 = 1/np.tan(OD1*np.pi/2)
        x1 = [D_intra, 0, kappa1]
        signal_ic1 = self.SynthMeasWatsonSHCylNeuman_PGSE(x1, n1)
        return signal_ic1

    def S_ec1(self, x):
        """
        This function models the extracellular component.
        The extra-cellular compartment refers to the space around the
        neurites, which is occupied by various types of glial cells and,
        additionally in gray matter, cell bodies (somas). In this space, the
        diffusion of water molecules is hindered by the presence of neurites
        but not restricted, hence is modeled with simple (Gaussian)
        anisotropic diffusion.
        (see [2]_ for a comparison and a thorough discussion)
        ----------
        References
        ----------
        .. [2] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
        OD1 = x[0]
        sinT1 = np.sin(x[1])
        cosT1 = np.cos(x[1])
        sinP1 = np.sin(x[2])
        cosP1 = np.cos(x[2])
        v_ic1 = x[3]
        n1 = [cosP1*sinT1, sinP1*sinT1, cosT1]
        kappa1 = 1/np.tan(OD1*np.pi/2)
        d_perp = D_intra * (1 - v_ic1)
        signal_ec1 = self.SynthMeasWatsonHinderedDiffusion_PGSE([D_intra,
                                                                 d_perp,
                                                                 kappa1], n1)
        return signal_ec1
<<<<<<< HEAD

    def S_ic2(self, x):
        """
        We extend the NODDI model as presented in [2]_ for two fiber
        orientations. Therefore we have 2 intracellular and extracellular
        components to account for this.

        S_ic2 corresponds to the second intracellular component in the NODDIx
        model

        (see Supplimentary note from 6: [1]_ for a comparison and a thorough
=======
        """
        We extend the NODDI model as presented in [2] for two fiber
        orientations. Therefore we have 2 intracellular and extracellular
        components to account for this.
        (see Supplimentary note 6: [1]_ for a comparison and a thorough
>>>>>>> refs/remotes/origin/nipy-dipy-master
        discussion)
        ----------
        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
<<<<<<< HEAD
        .. [2] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
=======
        """

    def S_ic2(self, x):
>>>>>>> refs/remotes/origin/nipy-dipy-master
        OD2 = x[4]
        sinT2 = np.sin(x[5])
        cosT2 = np.cos(x[5])
        sinP2 = np.sin(x[6])
        cosP2 = np.cos(x[6])
        n2 = [cosP2 * sinT2, sinP2 * sinT2, cosT2]
        kappa2 = 1/np.tan(OD2*np.pi/2)
        x2 = [D_intra, 0, kappa2]
        signal_ic2 = self.SynthMeasWatsonSHCylNeuman_PGSE(x2, n2)
        return signal_ic2

    def S_ec2(self, x):
<<<<<<< HEAD
        """
        We extend the NODDI model as presented in [2] for two fiber
        orientations. Therefore we have 2 extracellular and extracellular
        components to account for this.

        S_ic2 corresponds to the second extracellular component in the NODDIx
        model

        (see Supplimentary note 6: [1]_ for a comparison and a thorough
        discussion)
        ----------
        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
        .. [2] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
        OD2 = x[4]
        sinT2 = np.sin(x[5])
        cosT2 = np.cos(x[5])
        sinP2 = np.sin(x[6])
        cosP2 = np.cos(x[6])
        n2 = [cosP2 * sinT2, sinP2 * sinT2, cosT2]
        v_ic2 = x[7]
        d_perp2 = D_intra * (1 - v_ic2)
        kappa2 = 1/np.tan(OD2*np.pi/2)
        signal_ec2 = self.SynthMeasWatsonHinderedDiffusion_PGSE([D_intra,
                                                                 d_perp2,
                                                                 kappa2], n2)
        return signal_ec2

    def S_ec1_new(self, x, f):
<<<<<<< HEAD
        """
        This function is used in the second step of the MIX framework to
        construct the Phi when the data is fitted using the Differential
        Evolution. It is used to calculate the cost for non-linear least
        squares.

        Computes the extracellular component for the first fiber.

        Refer to the nlls_cost() function.
        """
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
        OD1 = x[0]
        sinT1 = np.sin(x[1])
        cosT1 = np.cos(x[1])
        sinP1 = np.sin(x[2])
        cosP1 = np.cos(x[2])
        v_ic1 = f[0]
        n1 = [cosP1*sinT1, sinP1*sinT1, cosT1]
        kappa1 = 1/np.tan(OD1*np.pi/2)
        d_perp = D_intra * (1 - v_ic1)
        signal_ec1 = self.SynthMeasWatsonHinderedDiffusion_PGSE([D_intra,
                                                                 d_perp,
                                                                 kappa1], n1)
        return signal_ec1

    def S_ec2_new(self, x, f):
<<<<<<< HEAD
        """
        This function is used in the second step of the MIX framework to
        construct the Phi when the data is fitted using the Differential
        Evolution. It is used to calculate the cost for non-linear least
        squares.

        Computes the extracellular component for the second fiber.

        Refer to the nlls_cost() function.
        """
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
        OD2 = x[3]
        sinT2 = np.sin(x[4])
        cosT2 = np.cos(x[4])
        sinP2 = np.sin(x[5])
        cosP2 = np.cos(x[5])
        n2 = [cosP2 * sinT2, sinP2 * sinT2, cosT2]
        v_ic2 = f[2]
        d_perp2 = D_intra * (1 - v_ic2)
        kappa2 = 1/np.tan(OD2*np.pi/2)
        signal_ec2 = self.SynthMeasWatsonHinderedDiffusion_PGSE([D_intra,
                                                                 d_perp2,
                                                                 kappa2], n2)
        return signal_ec2

    def S_ic2_new(self, x):
<<<<<<< HEAD
        """
        This function is used in the second step of the MIX framework to
        construct the Phi when the data is fitted using the Differential
        Evolution. It is used to calculate the cost for non-linear least
        squares.

        Computes the intracellular component for the second fiber.

        Refer to the nlls_cost() function.
        """
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
        OD2 = x[3]
        sinT2 = np.sin(x[4])
        cosT2 = np.cos(x[4])
        sinP2 = np.sin(x[5])
        cosP2 = np.cos(x[5])
        n2 = [cosP2 * sinT2, sinP2 * sinT2, cosT2]
        kappa2 = 1/np.tan(OD2*np.pi/2)
        x2 = [D_intra, 0, kappa2]
        signal_ic2 = self.SynthMeasWatsonSHCylNeuman_PGSE(x2, n2)
        return signal_ic2

<<<<<<< HEAD
    def SynthMeasWatsonSHCylNeuman_PGSE(self, x, fiberdir):
        """
        Substrate: Impermeable cylinders with one radius in an empty background
        Orientation distribution: Watson's distribution with SH approximation
        Pulse sequence: Pulsed gradient spin echo
        Signal approximation: Gaussian phase distribution.

        This returns the measurements E according to the model and the Jacobian
        J of the measurements with respect to the parameters.  The Jacobian
        does not include derivates with respect to the fibre direction.

        x is the list of model parameters in SI units:
            x(1) is the diffusivity of the material inside the cylinders.
            x(2) is the radius of the cylinders.
            x(3) is the concentration parameter of the Watson's distribution
        fibredir is a unit vector along the symmetry axis of the Watson's
        distribution.  It must be in Cartesian coordinates [x y z]' with size
        [3 1]. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
=======
    """

    """
    def SynthMeasWatsonSHCylNeuman_PGSE(self, x, fiberdir):
>>>>>>> refs/remotes/origin/nipy-dipy-master
        d = x[0]
        kappa = x[2]

        l_q = self.gtab.bvecs.shape[0]

        # parallel component
        LePar = self.CylNeumanLePar_PGSE(d)

        # Perpendicular component
<<<<<<< HEAD
=======
        # LePerp = CylNeumanLePerp_PGSE(d, R, G, delta, smalldel, roots)
>>>>>>> refs/remotes/origin/nipy-dipy-master
        LePerp = np.zeros((self.G.shape[0]))
        ePerp = np.exp(LePerp)

        # Compute the Legendre weighted signal
        Lpmp = LePerp - LePar
<<<<<<< HEAD

        # The Legendre Gauss Integran is computed from Cython
        # Please Refere: noddi_speed.pyx
        lgi = noddixspeed.legendre_gauss_integral(Lpmp, 6)

        # Compute the SH coefficients of the Watson's distribution
        coeff = noddixspeed.watson_sh_coeff(kappa)
=======
        lgi = self.LegendreGaussianIntegral(Lpmp, 6)

        # Compute the SH coefficients of the Watson's distribution
        coeff = self.WatsonSHCoeff(kappa)
>>>>>>> refs/remotes/origin/nipy-dipy-master
        coeffMatrix = np.tile(coeff, [l_q, 1])

        cosTheta = np.dot(self.gtab.bvecs, fiberdir)
        badCosTheta = np.where(abs(cosTheta) > 1)
<<<<<<< HEAD
        cosTheta[badCosTheta] = \
            cosTheta[badCosTheta] / abs(cosTheta[badCosTheta])
=======
        cosTheta[badCosTheta] = cosTheta[badCosTheta] / abs(cosTheta[badCosTheta])
>>>>>>> refs/remotes/origin/nipy-dipy-master

        # Compute the SH values at cosTheta
        sh = np.zeros(coeff.shape[0])
        shMatrix = np.tile(sh, [l_q, 1])

<<<<<<< HEAD
        # Computes a for loop for the Legendre matrix and evaulates the
        # Legendre Integral at a Point : Cython Code
        noddixspeed.synthMeasSHFor(cosTheta, shMatrix)
        E = np.sum(lgi * coeffMatrix * shMatrix, 1)
        E[np.isnan(E)] = 0.1
=======
        tmp = np.empty(cosTheta.shape)
        for i in range(7):
            shMatrix1 = np.sqrt((i + 1 - .75) / np.pi)
            noddixspeed.legendre_matrix(2 * (i + 1) - 2, cosTheta, tmp)
            shMatrix[:, i] = shMatrix1 * tmp

        E = np.sum(lgi * coeffMatrix * shMatrix, 1)
>>>>>>> refs/remotes/origin/nipy-dipy-master
        E[E <= 0] = min(E[E > 0]) * 0.1
        E = 0.5 * E * ePerp
        return E

    def CylNeumanLePar_PGSE(self, d):
<<<<<<< HEAD
        r"""
        Substrate: Parallel, impermeable cylinders with one radius in an empty
        background.
        Pulse sequence: Pulsed gradient spin echo
        Signal approximation: Gaussian phase distribution.

        This function returns the log signal attenuation in parallel direction
        (LePar) according to the Neuman model and the Jacobian J of LePar with
        respect to the parameters.  The Jacobian does not include derivates
        with respect to the fibre direction.

        d is the diffusivity of the material inside the cylinders.

        G, delta and smalldel are the gradient strength, pulse separation and
        pulse length of each measurement in the protocol. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
        # Radial wavenumbers
        modQ = gamma * self.small_delta * self.G
        modQ_Sq = modQ ** 2
        # Diffusion time for PGSE, in a matrix for the computation below.
        difftime = (self.big_delta - self.small_delta / 3)
<<<<<<< HEAD
=======

>>>>>>> refs/remotes/origin/nipy-dipy-master
        # Parallel component
        LE = -modQ_Sq * difftime * d
        return LE

<<<<<<< HEAD
    def SynthMeasWatsonHinderedDiffusion_PGSE(self, x, fibredir):
        """
        Substrate: Anisotropic hindered diffusion compartment
        Orientation distribution: Watson's distribution
        Pulse sequence: Pulsed gradient spin echo
        Signal approximation: N/A
        returns the measurements E according to the model and the Jacobian J of
        the measurements with respect to the parameters.  The Jacobian does not
        include derivates with respect to the fibre direction.

        x is the list of model parameters in SI units:
        x(0) is the free diffusivity of the material inside and outside the
        cylinders.
        x(1): is the hindered diffusivity outside the cylinders in
              perpendicular directions.
        x(2) is the concentration parameter of the Watson's distribution

        fibredir is a unit vector along the symmetry axis of the Watson's
        distribution. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
=======
    def LegendreGaussianIntegral(self, x, n):
        exact = np.where(x > 0.05)
        approx = np.where(x <= 0.05)
        mn = n + 1
        I = np.zeros((x.shape[0], mn))
        sqrtx = np.sqrt(x[exact])
        temp = np.empty(sqrtx.shape)
        noddixspeed.error_function(sqrtx, temp)
        I[exact, 0] = np.sqrt(np.pi) * temp / sqrtx
        dx = 1 / x[exact]
        emx = -np.exp(-x[exact])
        # here-----
        for i in range(2, mn + 1):
            I[exact, i - 1] = emx + (i - 1.5) * I[exact, i - 2]
            I[exact, i - 1] = I[exact, i - 1] * dx
#            b[:,1:] = b[:, :-1] * 10

        # Computing the legendre gaussian integrals for large enough x
        L = np.zeros((x.shape[0], n + 1))

        for i in range(0, n + 1):
            if i == 0:
                L[exact, 0] = I[exact, 0]
            elif i == 1:
                L[exact, 1] = -0.5 * I[exact, 0] + 1.5 * I[exact, 1]
            elif i == 2:
                L[exact, 2] = 0.375 * I[exact, 0] - 3.75 * I[exact, 1] \
                            + 4.375 * I[exact, 2]
            elif i == 3:
                L[exact, 3] = -0.3125 * I[exact, 0] + 6.5625 * I[exact, 1] \
                            - 19.6875 * I[exact, 2] + 14.4375 * I[exact, 3]
            elif i == 4:
                L[exact, 4] = 0.2734375 * I[exact, 0] \
                            - 9.84375 * I[exact, 1] \
                            + 54.140625 * I[exact, 2] \
                            - 93.84375 * I[exact, 3] \
                            + 50.2734375 * I[exact, 4]
            elif i == 5:
                L[exact, 5] = -(63. / 256) * I[exact, 0] \
                            + (3465. / 256) * I[exact, 1] \
                            - (30030. / 256) * I[exact, 2] \
                            + (90090. / 256) * I[exact, 3] \
                            - (109395. / 256) * I[exact, 4] \
                            + (46189. / 256) * I[exact, 5]
            elif i == 6:
                L[exact, 6] = (231. / 1024) * I[exact, 0] \
                            - (18018. / 1024)*I[exact, 1] \
                            + (225225. / 1024) * I[exact, 2] \
                            - (1021020. / 1024) * I[exact, 3] \
                            + (2078505. / 1024) * I[exact, 4] \
                            - (1939938. / 1024) * I[exact, 5] \
                            + (676039. / 1024) * I[exact, 6]

        # Computing the legendre gaussian integrals for small x
        x2 = pow(x[approx], 2)
        x3 = x2 * x[approx]
        x4 = x3 * x[approx]
        x5 = x4 * x[approx]
        x6 = x5 * x[approx]
        for i in range(0, n):
            if i == 0:
                L[approx, 0] = 2 - 2 * x[approx] / 3 + x2 / 5 - x3 / 21
                + x4 / 108
            elif i == 1:
                L[approx, 1] = -4 * x[approx] / 15 + 4 * x2 / 35
                - 2 * x3 / 63 + 2 * x4 / 297
            elif i == 2:
                L[approx, 2] = 8 * x2 / 315 - 8 * x3 / 693 + 4 * x4 / 1287
            elif i == 3:
                L[approx, 3] = -16 * x3 / 9009 + 16 * x4 / 19305
            elif i == 4:
                L[approx, 4] = 32 * x4 / 328185
            elif i == 5:
                L[approx, 5] = -64 * x5 / 14549535
            elif i == 6:
                L[approx, 6] = 128 * x6 / 760543875
        return L

    def WatsonSHCoeff(self, k):
        # The maximum order of SH coefficients (2n)
        n = 6
        # Computing the SH coefficients
        C = np.zeros((n + 1))
        # 0th order is a constant
        C[0] = 2 * np.sqrt(np.pi)

        # Precompute the special function values
        sk = np.sqrt(k)
        sk2 = sk * k
        sk3 = sk2 * k
        sk4 = sk3 * k
        sk5 = sk4 * k
        sk6 = sk5 * k
#        sk7 = sk6 * k[exact]
        k2 = k ** 2
        k3 = k2 * k
        k4 = k3 * k
        k5 = k4 * k
        k6 = k5 * k
#        k7 = k6 * k

        erfik = special.erfi(sk)
        ierfik = 1 / erfik
        ek = np.exp(k)
        dawsonk = 0.5 * np.sqrt(np.pi) * erfik / ek

        if k > 0.1:
            # for large enough kappa
            C[1] = 3 * sk - (3 + 2 * k) * dawsonk
            C[1] = np.sqrt(5) * C[1] * ek
            C[1] = C[1] * ierfik / k

            C[2] = (105 + 60 * k + 12 * k2) * dawsonk
            C[2] = C[2] - 105 * sk + 10 * sk2
            C[2] = .375 * C[2] * ek / k2
            C[2] = C[2] * ierfik

            C[3] = -3465 - 1890 * k - 420 * k2 - 40 * k3
            C[3] = C[3] * dawsonk
            C[3] = C[3] + 3465 * sk - 420 * sk2 + 84 * sk3
            C[3] = C[3] * np.sqrt(13 * np.pi) / 64 / k3
            C[3] = C[3] / dawsonk

            C[4] = 675675 + 360360 * k + 83160 * k2 + 10080 * k3 + 560 * k4
            C[4] = C[4] * dawsonk
            C[4] = C[4] - 675675 * sk + 90090 * sk2 - 23100 * sk3 + 744 * sk4
            C[4] = np.sqrt(17) * C[4] * ek
            C[4] = C[4] / 512 / k4
            C[4] = C[4] * ierfik

            C[5] = -43648605 - 22972950 * k - 5405400 * k2 - 720720 * k3 \
                - 55440 * k4 - 2016 * k5
            C[5] = C[5] * dawsonk
            C[5] = C[5] + 43648605 * sk - 6126120 * sk2 + 1729728 * sk3 \
                - 82368 * sk4 + 5104 * sk5
            C[5] = np.sqrt(21 * np.pi) * C[5] / 4096 / k5
            C[5] = C[5] / dawsonk

            C[6] = 7027425405 + 3666482820 * k + 872972100 * k2 \
                + 122522400 * k3 + 10810800 * k4 + 576576 * k5 + 14784 * k6
            C[6] = C[6] * dawsonk
            C[6] = C[6] - 7027425405 * sk + 1018467450 * sk2 \
                - 302630328 * sk3 + 17153136 * sk4 - 1553552 * sk5 \
                + 25376 * sk6
            C[6] = 5 * C[6] * ek
            C[6] = C[6] / 16384 / k6
            C[6] = C[6] * ierfik

        elif k > 30:
            # for very large kappa
            lnkd = np.log(k) - np.log(30)
            lnkd2 = lnkd * lnkd
            lnkd3 = lnkd2 * lnkd
            lnkd4 = lnkd3 * lnkd
            lnkd5 = lnkd4 * lnkd
            lnkd6 = lnkd5 * lnkd
            C[1] = 7.52308 + 0.411538 * lnkd - 0.214588 * lnkd2 \
                + 0.0784091 * lnkd3 - 0.023981 * lnkd4 + 0.00731537 * lnkd5 \
                - 0.0026467 * lnkd6
            C[2] = 8.93718 + 1.62147 * lnkd - 0.733421 * lnkd2 \
                + 0.191568 * lnkd3 - 0.0202906 * lnkd4 - 0.00779095 * lnkd5 \
                + 0.00574847*lnkd6
            C[3] = 8.87905 + 3.35689 * lnkd - 1.15935 * lnkd2 \
                + 0.0673053 * lnkd3 + 0.121857 * lnkd4 - 0.066642 * lnkd5 \
                + 0.0180215 * lnkd6
            C[4] = 7.84352 + 5.03178 * lnkd - 1.0193 * lnkd2 \
                - 0.426362 * lnkd3 + 0.328816 * lnkd4 - 0.0688176 * lnkd5 \
                - 0.0229398 * lnkd6
            C[5] = 6.30113 + 6.09914 * lnkd - 0.16088 * lnkd2 \
                - 1.05578 * lnkd3 + 0.338069 * lnkd4 + 0.0937157 * lnkd5 \
                - 0.106935 * lnkd6
            C[6] = 4.65678 + 6.30069 * lnkd + 1.13754 * lnkd2 \
                - 1.38393 * lnkd3 - 0.0134758 * lnkd4 + 0.331686 * lnkd5 \
                - 0.105954 * lnkd6

        elif k <= 0.1:
            # for small kappa
            C[1] = 4 / 3 * k + 8 / 63 * k2
            C[1] = C[1] * np.sqrt(np.pi / 5)

            C[2] = 8 / 21 * k2 + 32 / 693 * k3
            C[2] = C[2] * (np.sqrt(np.pi) * 0.2)

            C[3] = 16 / 693 * k3 + 32 / 10395 * k4
            C[3] = C[3] * np.sqrt(np.pi / 13)

            C[4] = 32 / 19305 * k4
            C[4] = C[4] * np.sqrt(np.pi / 17)

            C[5] = 64 * np.sqrt(np.pi / 21) * k5 / 692835

            C[6] = 128 * np.sqrt(np.pi) * k6 / 152108775
        return C

    def SynthMeasWatsonHinderedDiffusion_PGSE(self, x, fibredir):
>>>>>>> refs/remotes/origin/nipy-dipy-master
        dPar = x[0]
        dPerp = x[1]
        kappa = x[2]
        dw = self.WatsonHinderedDiffusionCoeff(dPar, dPerp, kappa)
        xh = [dw[0], dw[1]]
        E = self.SynthMeasHinderedDiffusion_PGSE(xh, fibredir)
        return E

    def WatsonHinderedDiffusionCoeff(self, dPar, dPerp, kappa):
<<<<<<< HEAD
        """
        Substrate: Anisotropic hindered diffusion compartment
        Orientation distribution: Watson's distribution
        WatsonHinderedDiffusionCoeff(dPar, dPerp, kappa)
        returns the equivalent parallel and perpendicular diffusion
        coefficients for hindered compartment with impermeable cylinder's
        oriented with a Watson's distribution with a cocentration parameter of
        kappa.

        dPar is the free diffusivity of the material inside and outside the
        cylinders.
        dPerp is the hindered diffusivity outside the cylinders in
        perpendicular directions.
        kappa is the concentration parameter of the Watson's distribution. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
        dw = np.zeros((2, 1))
        dParMdPerp = dPar - dPerp

        if kappa < 1e-5:
            dParP2dPerp = dPar + 2 * dPerp
            k2 = kappa * kappa
            dw[0] = dParP2dPerp / 3 + 4 * dParMdPerp * kappa / 45 \
                + 8 * dParMdPerp * k2 / 945
            dw[1] = dParP2dPerp / 3 - 2 * dParMdPerp * kappa / 45 \
                - 4 * dParMdPerp * k2 / 945
        else:
            sk = np.sqrt(kappa)
            dawsonf = 0.5 * np.exp(-kappa) * np.sqrt(np.pi) * special.erfi(sk)
            factor = sk / dawsonf
            dw[0] = (-dParMdPerp + 2 * dPerp * kappa +
                     dParMdPerp * factor) / (2*kappa)
            dw[1] = (dParMdPerp + 2 * (dPar + dPerp) * kappa -
                     dParMdPerp * factor) / (4 * kappa)
        return dw

    def SynthMeasHinderedDiffusion_PGSE(self, x, fibredir):
<<<<<<< HEAD
        """
        Substrate: Anisotropic hindered diffusion compartment
        Pulse sequence: Pulsed gradient spin echo
        Signal approximation: N/A

        This function returns the measurements E according to the model and the
        Jacobian J of the measurements with respect to the parameters. The
        Jacobian does not include derivates with respect to the fibre
        direction.

        x is the list of model parameters in SI units:
        x(0): is the free diffusivity of the material inside and outside the
             cylinders.
        x(1): is the hindered diffusivity outside the cylinders in
              perpendicular directions.

        fibredir is a unit vector along the cylinder axis. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
        dPar = x[0]
        dPerp = x[1]
        # Angles between gradient directions and fibre direction.
        cosTheta = np.dot(self.gtab.bvecs, fibredir)
        cosThetaSq = cosTheta ** 2
        # Find hindered signals
        E = np.exp(- self.gtab.bvals * ((dPar - dPerp) * cosThetaSq + dPerp))
        return E

    def x_f_to_x_and_f(self, x_f):
<<<<<<< HEAD
        """
        The MIX framework makes use of Variable Projections (VarPro) to
        separately fit the Volume Fractions and the other parameters that
        involve exponential functions.

        This function performs this task of taking the 11 input parameters of
        the signal and creates 2 separate lists:
            f: Volume  Fractions
            x: Other Signal Params [1]_

        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
        """
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
        f = np.zeros((1, 5))
        f = x_f[0:5]
        x = x_f[5:12]
        return x, f

    def x_and_f_to_x_f(self, x, f):
<<<<<<< HEAD
        """
        The MIX framework makes use of Variable Projections (VarPro) to
        separately fit the Volume Fractions and the other parameters that
        involve exponential functions.

        This function performs this task of taking the 11 input parameters of
        the signal and creates 2 separate lists:
            f: Volume  Fractions
            x: Other Signal Params [1]_

        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
        """
=======
>>>>>>> refs/remotes/origin/nipy-dipy-master
        x_f = np.zeros(11)
        f = np.squeeze(f)
        f11ga = x[3]
        f12ga = x[7]
        x_f[0] = (f[0] + f11ga) / 2
        x_f[1] = f[1]
        x_f[2] = (f[2] + f12ga) / 2
        x_f[3:5] = f[3:5]
        x_f[5:8] = x[0:3]
        x_f[8:11] = x[4:7]
<<<<<<< HEAD
        return x_f
=======
        return x_f

    def Phi(self, x):
        self.exp_phi1[:, 0] = self.S_ic1(x)
        self.exp_phi1[:, 1] = self.S_ec1(x)
        self.exp_phi1[:, 2] = self.S_ic2(x)
        self.exp_phi1[:, 3] = self.S_ec2(x)
        return self.exp_phi1

    def Phi2(self, x_f):
        x, f = self.x_f_to_x_and_f(x_f)
        self.exp_phi1[:, 0] = self.S_ic1(x)
        self.exp_phi1[:, 1] = self.S_ec1_new(x, f)
        self.exp_phi1[:, 2] = self.S_ic2_new(x)
        self.exp_phi1[:, 3] = self.S_ec2_new(x, f)
        return self.exp_phi1

    def estimate_signal(self, x_f):
        x, f = self.x_f_to_x_and_f(x_f)
        x1, x2 = self.x_to_xs(x)
        S = f[0] * self.S1_slow(x1) + f[1] * self.S2_slow(x2)
        + f[2] * self.S3() + f[3] * self.S4()
        return S

    def differential_evol_cost(self, phi, signal):

        """
        To make the cost function for differential evolution algorithm
        Parameters
        ----------
        phi:
            phi.shape = number of data points x 4
        signal:
            signal.shape = number of data points x 1
        Returns
        -------
        (signal -  S)^T(signal -  S)
        Notes
        --------
        to make cost function for differential evolution algorithm:
        .. math::
            (signal -  S)^T(signal -  S)
        """
        #  moore-penrose
        phi_mp = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
        #  sigma
        f = np.dot(phi_mp, signal)
        yhat = np.dot(phi, f)
        return np.dot((signal - yhat).T, signal - yhat)
>>>>>>> refs/remotes/origin/nipy-dipy-master
