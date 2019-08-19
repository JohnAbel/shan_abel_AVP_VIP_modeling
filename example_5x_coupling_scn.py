"""
Run model of the full SCN consisting of 60 Gonze oscillators. The period
is found using the deterministic model to normalize the resulting trajectories.
The initial conditions are random--these were hard-coded in to return an 
identical figure every time, but they can be commented out for random start.

Generates for final figure of example trajectories from the paper with a 
5:1 AVP:VIP coupling strength ratio.

John Abel
"""

from __future__ import division
import sys
assert sys.version[0]=='2', "This file must be run in python 2"

import numpy as np
import scipy as sp
import casadi as cs
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from local_imports import LimitCycle as lc
from local_imports import PlotOptions as plo
from local_models import gonze_model as ab
from local_models.gonze_model import y0in, param, ODEmodel, EqCount, ParamCount


#note: up to a point, this is the same as the file for the deterministic model!

# find original period
single_osc = lc.Oscillator(ODEmodel(), param, y0=np.ones(EqCount))
ts, sol = single_osc.int_odes(500)
single_osc.approx_y0_T(trans=2000)
wt_T = single_osc.T
single_osc.limit_cycle()

# number of each celltype
AVPcells = 20; VIPcells=20; NAVcells = 20
totcells = AVPcells+VIPcells+NAVcells

# initial phases
init_conditions_AV  = [single_osc.lc(wt_T*np.random.rand()) 
                        for i in range(AVPcells+VIPcells)]
init_conditions_NAV = [single_osc.lc(wt_T*np.random.rand())[:-1]
                        for i in range(NAVcells)]

#y0_random = np.hstack(init_conditions_AV+init_conditions_NAV)
# so that figure is identical use the one actually generated
y0_random = np.array(
      [0.03649647, 0.10663466, 2.60469301, 0.01192638, 0.15020802,
       0.19639209, 1.699136  , 0.04282985, 0.17498466, 0.22961959,
       1.65447964, 0.05062847, 0.27392133, 0.44823915, 1.69886225,
       0.09175747, 0.09678593, 0.5411219 , 3.46699745, 0.04373477,
       0.10162063, 0.14110837, 1.85774339, 0.02871033, 0.06361829,
       0.10819374, 2.11877411, 0.0185076 , 0.2658215 , 0.634519  ,
       2.10819409, 0.10519723, 0.03096468, 0.13758529, 2.97860106,
       0.01136174, 0.05646195, 0.1040406 , 2.20016264, 0.01667297,
       0.1366907 , 0.17977646, 1.73219853, 0.03876298, 0.19068233,
       0.69110311, 2.75665085, 0.08382894, 0.05829896, 0.37983743,
       3.60568227, 0.0258205 , 0.09417055, 0.53273561, 3.48273887,
       0.04253476, 0.03570413, 0.10808946, 2.63461023, 0.0117711 ,
       0.28179642, 0.56138241, 1.88543179, 0.10383703, 0.11665168,
       0.59565689, 3.33402542, 0.05275335, 0.11974919, 0.16030203,
       1.7844304 , 0.03382388, 0.12658435, 0.61753486, 3.26144801,
       0.05718728, 0.03193911, 0.17491349, 3.20998031, 0.01257403,
       0.04458828, 0.10106302, 2.38934048, 0.01373802, 0.03436042,
       0.20631866, 3.33939272, 0.01403256, 0.28076279, 0.49189016,
       1.75574768, 0.0974223 , 0.17125861, 0.67995127, 2.91303202,
       0.07621747, 0.09305787, 0.13268258, 1.90077683, 0.02635128,
       0.03093859, 0.13870795, 2.98752288, 0.01138396, 0.23254463,
       0.32751211, 1.62028786, 0.07131719, 0.07344282, 0.11536166,
       2.02984124, 0.02107731, 0.03316637, 0.19232516, 3.28694656,
       0.01334854, 0.16865091, 0.22075213, 1.66411195, 0.04858779,
       0.10453018, 0.14406946, 1.84449061, 0.02951992, 0.05530311,
       0.36288311, 3.6013551 , 0.0243991 , 0.03799234, 0.24170443,
       3.44350626, 0.01595947, 0.03255529, 0.11877973, 2.7951997 ,
       0.01127483, 0.04080215, 0.26524999, 3.49495685, 0.01737934,
       0.03095935, 0.13779727, 2.98029873, 0.01136584, 0.06170946,
       0.39814127, 3.60625163, 0.02743211, 0.18540888, 0.24484723,
       1.64112094, 0.05406722, 0.0372021 , 0.10556097, 2.5800791 ,
       0.01207008, 0.25520479, 0.6565869 , 2.21238126, 0.1035541 ,
       0.17709525, 0.23263695, 1.65152703, 0.05028992, 0.33242437,
       3.58392194, 0.06929097, 0.43548697, 3.59463344, 0.04627113,
       0.10100334, 2.35641368, 0.12577388, 0.16705929, 1.76430733,
       0.28085039, 0.57109287, 1.90849285, 0.0996584 , 0.54998659,
       3.44913621, 0.03121411, 0.16046703, 3.1335892 , 0.17179048,
       0.22511317, 1.65919141, 0.05515041, 0.36199537, 3.60102535,
       0.21525365, 0.29388882, 1.6198416 , 0.0493198 , 0.10139819,
       2.30291576, 0.06848032, 0.43169606, 3.59658506, 0.18030285,
       0.68627538, 2.84042064, 0.06126784, 0.10670654, 2.14364334,
       0.03377758, 0.11327042, 2.7216808 , 0.21964986, 0.30198587,
       1.61895939, 0.10876944, 0.57584794, 3.38915257, 0.14186617,
       0.18602192, 1.71872694, 0.28201589, 0.55856572, 1.87901187])



# switch to the multicellular stochastic model
from local_models.gonze_stoch_multi_params import param, GonzeModelManyCells

# note that these relative strengths only are about IN THE ABSENCE OF THE OTHER

# make the stochastic figure
model = GonzeModelManyCells(param, kav=5)
wt_trajectories = model.run(show_labels=False, seed=0)
wt_ts = wt_trajectories[0][:,0]
wt_avpsol = wt_trajectories[0][:,1:(80+1)]
wt_vipsol = wt_trajectories[0][:,(80+1):(160+1)]
wt_navsol = wt_trajectories[0][:,(160+1):]

# avp bmalko
avp_model = GonzeModelManyCells(param, kav=5, bmalko='AVP')
avp_trajectories = avp_model.run(show_labels=False, seed=0)
avp_ts = avp_trajectories[0][:,0]
avp_avpsol = avp_trajectories[0][:,1:(80+1)]
avp_vipsol = avp_trajectories[0][:,(80+1):(160+1)]
avp_navsol = avp_trajectories[0][:,(160+1):]

vip_model = GonzeModelManyCells(param, kav=5, bmalko='VIP')
vip_trajectories = vip_model.run(show_labels=False, seed=0)
vip_ts = vip_trajectories[0][:,0]
vip_avpsol = vip_trajectories[0][:,1:(80+1)]
vip_vipsol = vip_trajectories[0][:,(80+1):(160+1)]
vip_navsol = vip_trajectories[0][:,(160+1):]


# figure
plo.PlotOptions(ticks='in')
plt.figure(figsize=(3.5,3))
gs = gridspec.GridSpec(3,1)

ax = plt.subplot(gs[0,0])

ax.plot(wt_ts/wt_T, wt_vipsol[:,::4], 'darkorange', alpha=0.13, ls='--')
ax.plot(wt_ts/wt_T, wt_navsol[:,::3], 'goldenrod', alpha=0.13, ls='-.')
ax.plot(wt_ts/wt_T, wt_avpsol[:,::4], 'purple', alpha=0.13)
ax.plot(wt_ts/wt_T, wt_vipsol[:,::4].mean(1), 'darkorange', alpha=1, ls='--')
ax.plot(wt_ts/wt_T, wt_navsol[:,::3].mean(1), 'goldenrod', alpha=1, ls='-.')
ax.plot(wt_ts/wt_T, wt_avpsol[:,::4].mean(1), 'purple', alpha=1)

ax.set_xticks([0,1,2,3,4,5,6])
ax.set_xlim([0,6])
ax.set_xticklabels([])
ax.legend()
ax.set_ylim([0,300])
ax.set_yticks([0,100,200,300])

bx = plt.subplot(gs[1,0])
bx.plot(avp_ts/wt_T, avp_avpsol[:,::4], 'purple', alpha=0.13)
bx.plot(avp_ts/wt_T, avp_vipsol[:,::4], 'darkorange', alpha=0.13, ls='--')
bx.plot(avp_ts/wt_T, avp_navsol[:,::3], 'goldenrod', alpha=0.13, ls='-.')
bx.plot(avp_ts/wt_T, avp_avpsol[:,::4].mean(1), 'purple', alpha=1)
bx.plot(avp_ts/wt_T, avp_vipsol[:,::4].mean(1), 'darkorange', alpha=1, ls='--')
bx.plot(avp_ts/wt_T, avp_navsol[:,::3].mean(1), 'goldenrod', alpha=1, ls='-.')

bx.set_xticks([0,1,2,3,4,5,6])
bx.set_xticklabels([])
bx.set_xlim([0,6])
bx.set_ylim([0,300])
bx.set_yticks([0,100,200,300])


cx = plt.subplot(gs[2,0])
cx.plot(vip_ts/wt_T, vip_vipsol[:,::4], 'darkorange', alpha=0.13, ls='--')
cx.plot(vip_ts/wt_T, vip_navsol[:,::3], 'goldenrod', alpha=0.13, ls='-.')
cx.plot(vip_ts/wt_T, vip_avpsol[:,::4], 'purple', alpha=0.13)
cx.plot(vip_ts/wt_T, vip_vipsol[:,::4].mean(1), 'darkorange', alpha=1, ls='--')
cx.plot(vip_ts/wt_T, vip_navsol[:,::3].mean(1), 'goldenrod', alpha=1, ls='-.')
cx.plot(vip_ts/wt_T, vip_avpsol[:,::4].mean(1), 'purple', alpha=1)

cx.set_xticks([0,1,2,3,4,5,6])
cx.set_xlim([0,6])
cx.set_ylim([0,300])
cx.set_yticks([0,100,200,300])
cx.set_xlabel('Day')

plt.legend()
plt.tight_layout(**plo.layout_pad)
plt.savefig('results/model_figure_data.svg')
plt.show()

# # export data
# header = ['TimesH', 'Frame']+['AVP'+str(i) for i in range(AVPcells)]+['VIP'+str(i) for i in range(VIPcells)]+['NAV'+str(i) for i in range(NAVcells)]


# # wt traces
# output_data = np.hstack([np.array([wt_ts*24/wt_T]).T,  np.array([np.arange(len(wt_ts))]).T, wt_avpsol[:,::4], wt_vipsol[:,::4], wt_navsol[:,::3] ])
# output_df = pd.DataFrame(data=output_data,
#             columns = header)
# output_df.to_csv('Results/WT_stochastic_resync.csv', index=False)

# # avp traces
# output_data = np.hstack([np.array([avp_ts*24/wt_T]).T, np.array([np.arange(len(avp_ts))]).T, avp_avpsol[:,::4], avp_vipsol[:,::4], avp_navsol[:,::3] ])
# output_df = pd.DataFrame(data=output_data,
#             columns = header)
# output_df.to_csv('Results/AVPBmalKO_stochastic_resync.csv', index=False)

# # vip traces
# output_data = np.hstack([np.array([vip_ts*24/wt_T]).T, np.array([np.arange(len(vip_ts))]).T, vip_avpsol[:,::4], vip_vipsol[:,::4], vip_navsol[:,::3] ])
# output_df = pd.DataFrame(data=output_data,
#             columns = header)
# output_df.to_csv('Results/VIPBmalKO_stochastic_resync.csv', index=False)






















