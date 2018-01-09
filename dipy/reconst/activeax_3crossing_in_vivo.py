# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:28:39 2017

@author: mafzalid
"""
from dipy.reconst.base import ReconstModel
import numpy as np
import cvxpy as cvx
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
from dipy.reconst.recspeed import S1, S2, S2_new

gamma = 2.675987 * 10 ** 8

am = np.array([1.84118307861360, 5.33144196877749,
               8.53631578218074, 11.7060038949077,
               14.8635881488839, 18.0155278304879,
               21.1643671187891, 24.3113254834588,
               27.4570501848623, 30.6019229722078,
               33.7461812269726, 36.8899866873805,
               40.0334439409610, 43.1766274212415,
               46.3195966792621, 49.4623908440429,
               52.6050411092602, 55.7475709551533,
               58.8900018651876, 62.0323477967829,
               65.1746202084584, 68.3168306640438,
               71.4589869258787, 74.6010956133729,
               77.7431620631416, 80.8851921057280,
               84.0271895462953, 87.1691575709855,
               90.3110993488875, 93.4530179063458,
               96.5949155953313, 99.7367932203820,
               102.878653768715, 106.020498619541,
               109.162329055405, 112.304145672561,
               115.445950418834, 118.587744574512,
               121.729527118091, 124.871300497614,
               128.013065217171, 131.154821965250,
               134.296570328107, 137.438311926144,
               140.580047659913, 143.721775748727,
               146.863498476739, 150.005215971725,
               153.146928691331, 156.288635801966,
               159.430338769213, 162.572038308643,
               165.713732347338, 168.855423073845,
               171.997111729391, 175.138794734935,
               178.280475036977, 181.422152668422,
               184.563828222242, 187.705499575101])
# am = np.array([1.84118307861360])


class ActiveAxModel(ReconstModel):

    def __init__(self, gtab, params, D_intra, D_iso, fit_method='MIX'):
        r""" MIX framework (MIX) [1]_.

        The MIX computes the ActiveAx parameters. ActiveAx is a multi
        compartment model, (sum of exponentials).
        This algorithm uses three different optimizer. It starts with a
        differential evolutionary algorithm and fits the parameters in the
        power of exponentials. Then the fitted parameters in the first step are
        utilized to make a linear convex problem. Using a convex optimization,
        the volume fractions are determined. Then the last step is non linear
        least square fitting on all the parameters. The results of the first
        and second step are utilized as the initial values for the last step
        of the algorithm. (see [1]_ for a comparison and a through discussion).

        Parameters
        ----------
        gtab : GradientTable

        fit_method : str or callable

        Returns
        -------
        ActiveAx parameters

        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).

        """

        self.maxiter = 1000  # The maximum number of generations, genetic
#        algorithm 1000 default, 1
        self.xtol = 1e-8  # Tolerance for termination, nonlinear least square
#        1e-8 default, 1e-3
        self.D_intra = D_intra
        self.D_iso = D_iso
        self.gtab = gtab
        self.big_delta = gtab.big_delta
        self.small_delta = gtab.small_delta
        self.gamma = gamma
        self.G = params[:, 3] / 10 ** 6  # gradient strength
        self.G2 = self.G ** 2
        self.yhat_ball = self.D_iso * self.gtab.bvals
        self.L = self.gtab.bvals * self.D_intra
           
        self.yhat_zeppelin1 = np.zeros(self.small_delta.shape[0])
        self.yhat_zeppelin2 = np.zeros(self.small_delta.shape[0])
        self.yhat_zeppelin3 = np.zeros(self.small_delta.shape[0])
        self.yhat_cylinder1 = np.zeros(self.small_delta.shape[0])
        self.yhat_cylinder2 = np.zeros(self.small_delta.shape[0])
        self.yhat_cylinder3 = np.zeros(self.small_delta.shape[0])
        self.exp_phi1 = np.zeros((self.small_delta.shape[0], 7))
        self.exp_phi1[:, 4] = np.exp(-self.yhat_ball)
        self.x2 = np.zeros(9)

    def fit(self, data):
        """ Fit method of the ActiveAx model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        """
        
        bounds = np.array([(0.011, np.pi), (0.011, np.pi), (0.11, 15.9),
                          (0.11, 0.8), (0.011, np.pi), (0.011, np.pi),
                          (0.11, 15.9), (0.11, 0.8), (0.011, np.pi),
                          (0.011, np.pi), (0.11, 15.9), (0.11, 0.8)])
    
        res_one = differential_evolution(self.stoc_search_cost, bounds,
                                             maxiter=self.maxiter,
                                             args=(data,))
        x = res_one.x
        phi = self.Phi(x)
        fe = self.cvx_fit(data, phi)
        x_fe = self.x_and_fe_to_x_fe(x, fe)
        bounds = ([0.01, 0.01,  0, 0, 0.01, 0.01,  0.01, 0.01, 0, 0, 0.01,
                       0.01, 0.01, 0, 0, 0.01], [0.9,  0.9, np.pi, np.pi, 16,
                                            0.9, 0.9, 0.9, np.pi, np.pi, 16,
                                            0.9, 0.9, np.pi, np.pi, 16])
        res = least_squares(self.nlls_cost, x_fe, bounds=(bounds),
                                xtol=self.xtol, args=(data,))
        result = res.x
        return result

    def stoc_search_cost(self, x, signal):
        """
        Aax_exvivo_nlin

        Cost function for genetic algorithm

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

        Returns
        -------
        (signal -  S)^T(signal -  S)

        Notes
        --------
        cost function for genetic algorithm:

        .. math::

            (signal -  S)^T(signal -  S)
        """
        phi = self.Phi(x)
        return self.activeax_cost_one(phi, signal)

    def activeax_cost_one(self, phi, signal):  # sigma

        """
        Aax_exvivo_nlin
        to make cost function for genetic algorithm
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
        to make cost function for genetic algorithm:
        .. math::
            (signal -  S)^T(signal -  S)
        """

        phi_mp = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)  # moore-penrose
        f = np.dot(phi_mp, signal)
        yhat = np.dot(phi, f)  # - sigma
        return np.dot((signal - yhat).T, signal - yhat)

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
        f1, f2, f3, f4 (volume fractions)
        f1 = fe[0]
        f2 = fe[1]
        f3 = fe[2]
        f4 = fe[3]

        Notes
        --------
        cost function for genetic algorithm:

        .. math::

            minimize(norm((signal)- (phi*fe)))
        """

        # Create four scalar optimization variables.
        fe = cvx.Variable(7)
        # Create four constraints.
        constraints = [cvx.sum_entries(fe) == 1,
                       fe[0] >= 0.011,
                       fe[1] >= 0.011,
                       fe[2] >= 0.011,
                       fe[3] >= 0.011,
                       fe[4] >= 0.011,
                       fe[5] >= 0.011,
                       fe[6] >= 0.011,
                       fe[0] <= 0.89,
                       fe[1] <= 0.89,
                       fe[2] <= 0.89,
                       fe[3] <= 0.89,
                       fe[4] <= 0.89,
                       fe[5] <= 0.89,
                       fe[6] <= 0.89]

        # Form objective.
        obj = cvx.Minimize(cvx.sum_entries(cvx.square(phi * fe - signal)))

        # Form and solve problem.
        prob = cvx.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        return np.array(fe.value)

    def nlls_cost(self, x_fe, signal):
        """
        cost function for the least square problem

        Parameters
        ----------
        x_fe : array
            x_fe(0) x_fe(1)  are f1 f2 
            x_fe(2) theta1
            x_fe(3) phi1
            x_fe(4) R1
            x_fe(5) f3
            x_fe(6) f4
            x_fe(7) f5
            x_fe(8) theta2
            x_fe(9) phi2
            x_fe(10) R2

        signal_param : array
            signal_param.shape = number of data points x 7

            signal_param = np.hstack([signal[:, None], bvals[:, None], bvecs,
                                  G[:, None], small_delta[:, None],
                                  big_delta[:, None]])

        Returns
        -------
        sum{(signal -  phi*fe)^2}

        Notes
        --------
        cost function for the least square problem

        .. math::

            sum{(signal -  phi*fe)^2}
        """

        x, fe = self.x_fe_to_x_and_fe(x_fe)
        phi = self.Phi2(x_fe)
        return np.sum((np.dot(phi, fe) - signal) ** 2)

    def x_to_xs(self, x):
        x1 = np.zeros(9)
        x1[0:3] = x[0:3]
        x1[3:6] = x[4:7]
        x1[6:9] = x[8:11]
        self.x2[0:2] = x[0:2]
        self.x2[2] = x[3]
        self.x2[3:5] = x[4:6]
        self.x2[5] = x[7]
        self.x2[6:8] = x[8:10]
        self.x2[8] = x[11]
        return x1, self.x2

    def x_fe_to_x_and_fe(self, x_fe):
        x = np.zeros(9)
        fe = np.zeros(7)
        fe[:2] = x_fe[:2]
        x[0:3] = x_fe[2:5]
        fe[2:5] = x_fe[5:8]
        x[3:6] = x_fe[8:11]
        fe[5:7] = x_fe[11:13]
        x[6:9] = x_fe[13:16]
        return x, fe
      
    def x_and_fe_to_x_fe(self, x, fe):
        x_fe = np.zeros(16)
        fe = fe[:, 0]
        x_fe[:2] = fe[:2]
        x_fe[2:5] = x[0:3]
        x_fe[5:8] = fe[2:5]
        x_fe[8:11] = x[4:7]
        x_fe[11:13] = fe[5:7]
        x_fe[13:16] = x[8:11]
        return x_fe

    def Phi(self, x):
        x1, self.x2 = self.x_to_xs(x)
        S2(self.x2[0:3], self.gtab.bvals, self.gtab.bvecs, self.D_intra,
           self.yhat_zeppelin1)
        S2(self.x2[3:6], self.gtab.bvals, self.gtab.bvecs, self.D_intra,
           self.yhat_zeppelin2)
        S2(self.x2[6:9], self.gtab.bvals, self.gtab.bvecs, self.D_intra,
           self.yhat_zeppelin3)
        S1(x1[0:3], am, self.gtab.bvecs, self.gtab.bvals, self.small_delta,
           self.big_delta, self.G2, self.L, self.D_intra, self.yhat_cylinder1)
        S1(x1[3:6], am, self.gtab.bvecs, self.gtab.bvals, self.small_delta,
           self.big_delta, self.G2, self.L, self.D_intra, self.yhat_cylinder2)
        S1(x1[6:9], am, self.gtab.bvecs, self.gtab.bvals, self.small_delta,
           self.big_delta, self.G2, self.L, self.D_intra, self.yhat_cylinder3)
        self.exp_phi1[:, 0] = np.exp(-self.yhat_cylinder1)
        self.exp_phi1[:, 1] = np.exp(-self.yhat_zeppelin1)
        self.exp_phi1[:, 2] = np.exp(-self.yhat_cylinder2)
        self.exp_phi1[:, 3] = np.exp(-self.yhat_zeppelin2)
        self.exp_phi1[:, 5] = np.exp(-self.yhat_cylinder3)
        self.exp_phi1[:, 6] = np.exp(-self.yhat_zeppelin3)
        return self.exp_phi1

    def Phi2(self, x_fe):
        x, fe = self.x_fe_to_x_and_fe(x_fe)
        S2_new(x[0:2], fe[0:2], self.gtab.bvals, self.gtab.bvecs, self.D_intra,
               self.yhat_zeppelin1)
        S2_new(x[3:5], fe[2:5], self.gtab.bvals, self.gtab.bvecs, self.D_intra,
               self.yhat_zeppelin2)
        S2_new(x[6:8], fe[6:8], self.gtab.bvals, self.gtab.bvecs, self.D_intra,
               self.yhat_zeppelin3)
        S1(x[0:3], am, self.gtab.bvecs, self.gtab.bvals, self.gtab.small_delta,
           self.gtab.big_delta, self.G2, self.L, self.D_intra,
           self.yhat_cylinder1)
        S1(x[3:6], am, self.gtab.bvecs, self.gtab.bvals, self.gtab.small_delta,
           self.gtab.big_delta, self.G2, self.L, self.D_intra,
           self.yhat_cylinder2)
        S1(x[6:9], am, self.gtab.bvecs, self.gtab.bvals, self.gtab.small_delta,
           self.gtab.big_delta, self.G2, self.L, self.D_intra,
           self.yhat_cylinder3)
        self.exp_phi1[:, 0] = np.exp(-self.yhat_cylinder1)
        self.exp_phi1[:, 1] = np.exp(-self.yhat_zeppelin1)
        self.exp_phi1[:, 2] = np.exp(-self.yhat_cylinder2)
        self.exp_phi1[:, 3] = np.exp(-self.yhat_zeppelin2)
        self.exp_phi1[:, 5] = np.exp(-self.yhat_cylinder3)
        self.exp_phi1[:, 6] = np.exp(-self.yhat_zeppelin3)
        return self.exp_phi1