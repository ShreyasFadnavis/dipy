# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:17:00 2017

@author: mafzalid
"""
from scipy.optimize import differential_evolution
import dipy.reconst.NODDIx as NODDIx

fit_method = 'MIX'
NODDIx_model = NODDIx.NODDIxModel(gtab, params,
                                  fit_method=fit_method)
data = signal
res_one = differential_evolution(NODDIx_model.stoc_search_cost, bounds,
                                         maxiter=NODDIx_model.maxiter, args=(data,))

x = res_one.x

phi = NODDIx_model.Phi(x)
fe = NODDIx_model.cvx_fit(data, phi)

x_fe = NODDIx_model.x_and_fe_to_x_fe(x, fe)
# x = np.array([0.0369, 1.3581, 3.0141, 0.4424, 0.3418, 0.9324, 1.3446, 0.1454])

NODDIx_model.stoc_search_cost(x, signal)

signal_ic1 = NODDIx_model.S_ic1(x)

NODDIx_model.G[287] 

gtab.big_delta[286]

l_q = NODDIx_model.gtab.bvecs.shape[0]
x = [D_intra, 0, kappa1]
d = x[0]
LePar = NODDIx_model.CylNeumanLePar_PGSE(d)


LePerp = np.zeros((NODDIx_model.G.shape[0]))

lgi = NODDIx_model.LegendreGaussianIntegral(Lpmp, 6)

coeff = NODDIx_model.WatsonSHCoeff(kappa)

x = np.array([0.5172, 0.1727, 0.1697, 0.0100, 0.1285, 0.0562, 1.2983, 2.9486, 
              0.1401, 1.1232, 0.7282])
signal_ic1 = NODDIx_model.S_ic1(x1)
signal_ec1 = NODDIx_model.S_ec1_new(x1, fe)
signal_ic2 = NODDIx_model.S_ic2_new(x1)
signal_ec2 = NODDIx_model.S_ec2_new(x1, fe)
x1, fe = NODDIx_model.x_fe_to_x_and_fe(x_fe)
phi = NODDIx_model.Phi2(x_fe)
f = np.dot(phi, fe)  
S5 = NODDIx_model.exp_phi1[:, 4] 
nG7 = NODDIx_model.yhat_ball
S5 = np.exp(-nG7)

S5 = np.exp(-NODDIx_model.yhat_ball)

x1 = np.array([0.0562, 1.2983, 2.9486, 0.5172, 0.1401, 1.1232, 0.7282, 0.1697])
phi = NODDIx_model.Phi(x1)

cost = NODDIx_model.stoc_search_cost(x1, signal)

