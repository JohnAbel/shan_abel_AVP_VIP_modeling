"""
Run model of the full SCN consisting of 60 Gonze oscillators. The period
is found using the deterministic model to normalize the resulting trajectories.
The initial conditions are random--these were hard-coded in to return an 
identical figure every time, but they can be commented out for random start.

Generates for final figure of example trajectories from the paper with a 
2.5:1 AVP:VIP coupling strength ratio, and a 2:1 AVP:VIP cell count ratio.

John Abel
"""

from __future__ import division
import sys
assert sys.version[0]=='2', "This file must be run in python 2"
import pickle
from itertools import combinations

import numpy as np
import scipy as sp
import casadi as cs
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import minepy as mp
from scipy import stats

from local_imports import LimitCycle as lc
from local_imports import PlotOptions as plo
from local_models.gonze_model import y0in, param, ODEmodel, EqCount


#note: up to a point, this is the same as the file for the deterministic model!

# find original period
single_osc = lc.Oscillator(ODEmodel(), param, y0=np.ones(EqCount))
ts, sol = single_osc.int_odes(500)
single_osc.approx_y0_T(trans=2000)
wt_T = single_osc.T
single_osc.limit_cycle()

# number of each celltype
# these and the kav are hard-coded into the model
AVPcells = 53; VIPcells=27; NAVcells = 170
totcells = AVPcells+VIPcells+NAVcells

# initial phases
init_conditions_AV  = [single_osc.lc(wt_T*np.random.rand()) 
                        for i in range(AVPcells+VIPcells)]
init_conditions_NAV = [single_osc.lc(wt_T*np.random.rand())[:-1]
                        for i in range(NAVcells)]

# y0_random = np.hstack(init_conditions_AV+init_conditions_NAV)
# so that figure is identical use the one actually generated
y0_random = np.array([
       0.10043143, 0.12444713, 1.85956553, 0.02763109, 0.03308952,
       0.22212815, 4.18096   , 0.01400979, 0.07484931, 0.48911421,
       4.39234191, 0.03306732, 0.27183353, 0.38402442, 1.62543083,
       0.08493034, 0.02318264, 0.1200341 , 3.63656901, 0.0090209 ,
       0.31233276, 0.57931122, 1.86344863, 0.11240526, 0.14665215,
       0.72282171, 3.79538166, 0.0654036 , 0.02822243, 0.07053194,
       2.78929573, 0.00899138, 0.05868274, 0.40210256, 4.42963539,
       0.02573233, 0.02518812, 0.07474101, 2.96479526, 0.00837976,
       0.31072304, 0.54650071, 1.80374368, 0.1090757 , 0.03970584,
       0.27508187, 4.31279971, 0.01707716, 0.31230514, 0.57771086,
       1.86030898, 0.11225782, 0.06285979, 0.42622177, 4.42788671,
       0.02762874, 0.02433179, 0.07720182, 3.03460027, 0.00823818,
       0.30599822, 0.50731013, 1.74403977, 0.10432669, 0.03193974,
       0.06924584, 2.64321608, 0.00983495, 0.29946031, 0.4731876 ,
       1.70116176, 0.09958704, 0.0239272 , 0.07878342, 3.07382139,
       0.00818151, 0.02618672, 0.07280138, 2.89784157, 0.00856755,
       0.02445185, 0.07679418, 3.02386922, 0.0082565 , 0.25088995,
       0.78134162, 2.71485381, 0.10710108, 0.08355043, 0.52948107,
       4.34620088, 0.03701164, 0.1521263 , 0.73302977, 3.73992285,
       0.06782365, 0.10648672, 0.61792723, 4.17567408, 0.04739399,
       0.2105337 , 0.2645962 , 1.61189045, 0.061025  , 0.03529169,
       0.06963391, 2.5432656 , 0.01063353, 0.09019261, 0.1139381 ,
       1.91148734, 0.02485001, 0.15879203, 0.74428083, 3.67181646,
       0.07075346, 0.11800664, 0.14336723, 1.78807033, 0.03250341,
       0.03628457, 0.06991754, 2.51747836, 0.01087427, 0.03482013,
       0.06952222, 2.55605084, 0.01051977, 0.02397526, 0.07857621,
       3.06888055, 0.00818777, 0.13802899, 0.16627751, 1.72635383,
       0.03822461, 0.27678469, 0.74938285, 2.4381663 , 0.11436022,
       0.1245817 , 0.67215805, 4.01212157, 0.05554927, 0.17384931,
       0.76514732, 3.5164083 , 0.07728737, 0.03138495, 0.20740142,
       4.13155782, 0.01320721, 0.02280988, 0.11304308, 3.57070789,
       0.0087597 , 0.10229105, 0.60351153, 4.2106989 , 0.04549733,
       0.02261766, 0.10841957, 3.52309963, 0.00860012, 0.27309276,
       0.75552966, 2.47867378, 0.11347943, 0.04498153, 0.3136277 ,
       4.37353369, 0.01949505, 0.23907151, 0.78873526, 2.83815889,
       0.10313866, 0.13484968, 0.1625396 , 1.73501121, 0.03730293,
       0.04744984, 0.07592023, 2.29913242, 0.01365486, 0.13293764,
       0.16031024, 1.74040966, 0.03675113, 0.13672543, 0.1647402 ,
       1.72985618, 0.03784607, 0.02244707, 0.10209461, 3.45161816,
       0.00840402, 0.13494368, 0.16264955, 1.7347496 , 0.0373301 ,
       0.16660686, 0.20175224, 1.66406119, 0.04676488, 0.29548565,
       0.45664151, 1.68320818, 0.09710417, 0.15218904, 0.73314165,
       3.73928448, 0.06785131, 0.04456397, 0.31067843, 4.3697656 ,
       0.0193042 , 0.16453781, 0.19906157, 1.66773323, 0.04612977,
       0.03133275, 0.20694146, 4.12990736, 0.0131825 , 0.08845987,
       0.11219894, 1.92121308, 0.02438327, 0.11297515, 0.6388091 ,
       4.11909568, 0.05032353, 0.02544378, 0.14965804, 3.85742229,
       0.0102977 , 0.28417565, 0.41804394, 1.64825535, 0.09088071,
       0.02590373, 0.15473165, 3.88824126, 0.01053528, 0.20189973,
       0.78827295, 3.22462144, 0.08902176, 0.03742043, 0.25744251,
       4.27586418, 0.0160237 , 0.2905607 , 0.7197325 , 2.27998471,
       0.11696992, 0.30458762, 0.6697864 , 2.09025829, 0.11767262,
       0.29872532, 0.69461616, 2.17607197, 0.11776173, 0.09949355,
       0.12346843, 1.86396186, 0.02737466, 0.06864858, 0.09324821,
       2.05873069, 0.01912333, 0.04767415, 0.3322575 , 4.39426146,
       0.02072421, 0.24729056, 0.32935934, 1.60516464, 0.07453877,
       0.20363475, 0.78905762, 3.20656219, 0.08972311, 0.29931279,
       0.47253195, 1.70041579, 0.09949087, 0.28608906, 0.42394285,
       1.65297555, 0.09186933, 0.28572562, 0.42280636, 1.65204873,
       0.09167988, 0.04508407, 0.31434941, 4.37443498, 0.01954191,
       0.02321349, 0.08280441, 3.16059   , 0.00811164, 0.12262994,
       0.14852676, 1.77219835, 0.03380744, 0.06567672, 0.44181555,
       4.42320188, 0.02890703, 0.1391637 , 0.70738776, 3.87034939,
       0.06207558, 0.04804469, 0.33477009, 4.39666421, 0.02089318,
       0.03829721, 0.26428325, 4.29093733, 0.06109731, 0.41619324,
       4.42943418, 0.15457169, 0.18638076, 1.68714802, 0.13869949,
       0.16707082, 1.72457752, 0.28099586, 0.74155201, 2.39119885,
       0.13387856, 0.69543251, 3.92247532, 0.03351122, 0.22569092,
       4.19196521, 0.0242242 , 0.07758903, 3.04454113, 0.03317582,
       0.22285971, 4.18324846, 0.16213411, 0.74944799, 3.63747929,
       0.20109269, 0.78788271, 3.23302247, 0.26113907, 0.35858408,
       1.61337982, 0.17573701, 0.2138797 , 1.64924517, 0.02241325,
       0.09693131, 3.38658151, 0.02539822, 0.07426632, 2.94964944,
       0.10979716, 0.13439321, 1.81907753, 0.21537276, 0.79242214,
       3.08446266, 0.02316591, 0.119751  , 3.63402994, 0.0773472 ,
       0.10134031, 1.99156947, 0.02303456, 0.08425271, 3.18845759,
       0.03690195, 0.25335221, 4.26637617, 0.08130733, 0.10514863,
       1.96477177, 0.02550189, 0.07404673, 2.94240605, 0.23066133,
       0.29820095, 1.60369507, 0.31236126, 0.59212212, 1.88949106,
       0.02331061, 0.12213504, 3.65510141, 0.03929994, 0.27199375,
       4.3067818 , 0.07259099, 0.4779419 , 4.40187641, 0.0627244 ,
       0.42545891, 4.42804499, 0.17669879, 0.21518234, 1.64781232,
       0.0237755 , 0.12906546, 3.71264557, 0.12615552, 0.15251312,
       1.76079183, 0.17234767, 0.76334068, 3.53197469, 0.2080751 ,
       0.79072865, 3.16035644, 0.03250782, 0.2171632 , 4.16502666,
       0.09649588, 0.5823585 , 4.25666305, 0.04941067, 0.07730262,
       2.26995146, 0.10221955, 0.60325945, 4.21128384, 0.02267118,
       0.10983478, 3.53805198, 0.04096862, 0.0719362 , 2.41254487,
       0.0309522 , 0.20357125, 4.11760378, 0.0355173 , 0.24225534,
       4.23875031, 0.02996443, 0.06961349, 2.71470646, 0.05113903,
       0.35529086, 4.41297407, 0.2209922 , 0.2815782 , 1.60636307,
       0.25948327, 0.77337212, 2.62447947, 0.06135776, 0.41768862,
       4.42927693, 0.23977235, 0.31482051, 1.60348623, 0.02465588,
       0.1404774 , 3.79712924, 0.29938702, 0.47286139, 1.70079025,
       0.02614409, 0.15731758, 3.90332646, 0.02252633, 0.09139906,
       3.30827976, 0.02256004, 0.09058333, 3.29581297, 0.07943547,
       0.51090622, 4.36972851, 0.04210861, 0.07255653, 2.39035147,
       0.03850088, 0.26585895, 4.29427216, 0.05240518, 0.36345762,
       4.41788015, 0.30730763, 0.51602237, 1.7563055 , 0.14724094,
       0.72396252, 3.78944044, 0.2503662 , 0.78175391, 2.72033702,
       0.23075997, 0.79166102, 2.92457324, 0.0299143 , 0.06963089,
       2.71668157, 0.02370881, 0.1281235 , 3.70512706, 0.119004  ,
       0.14447378, 1.7845549 , 0.11589924, 0.64768195, 4.09277175,
       0.02904007, 0.18611485, 4.04748622, 0.02301852, 0.11715371,
       3.61025082, 0.18139492, 0.77333286, 3.43804943, 0.10229594,
       0.60352878, 4.21065884, 0.03664735, 0.25133098, 4.26155243,
       0.22173017, 0.28281377, 1.60607497, 0.11004196, 0.62957646,
       4.145006  , 0.1834552 , 0.77531236, 3.41662402, 0.03808473,
       0.26263413, 4.28739273, 0.02502512, 0.07513924, 2.97701222,
       0.31160139, 0.61395331, 1.9378747 , 0.18388163, 0.77570851,
       3.41218839, 0.03149802, 0.06928005, 2.65822732, 0.08122353,
       0.51908833, 4.35985252, 0.30983514, 0.63521044, 1.99047517,
       0.02429035, 0.0773484 , 3.03839125, 0.31156433, 0.61457644,
       1.93933548, 0.15103847, 0.73107176, 3.7509813 , 0.066682  ,
       0.44725448, 4.42092253, 0.0704444 , 0.46704203, 4.40984109,
       0.02904631, 0.0700109 , 2.75239269, 0.03531963, 0.06964102,
       2.54251937, 0.06413275, 0.0892208 , 2.09911475, 0.24532145,
       0.78531555, 2.77304734, 0.06813508, 0.09278367, 2.06311039,
       0.12827555, 0.68175727, 3.97685587, 0.04049385, 0.07169045,
       2.42212584, 0.24143313, 0.31796325, 1.60370629, 0.02846308,
       0.07036141, 2.77817947, 0.18217516, 0.2226956 , 1.64010342,
       0.22433708, 0.79270093, 2.9913045 , 0.09685234, 0.58370252,
       4.25392553, 0.02357154, 0.08052714, 3.11339876, 0.20544802,
       0.25666745, 1.61553426, 0.02805626, 0.07065941, 2.79714763,
       0.23363257, 0.79085431, 2.8947203 , 0.02280692, 0.11297951,
       3.57007681, 0.12837705, 0.68201442, 3.97587979, 0.08077573,
       0.1046333 , 1.96824926, 0.02899481, 0.18568972, 4.04563441,
       0.02268483, 0.08828036, 3.2590967 , 0.02689509, 0.16515894,
       3.94669838, 0.24889729, 0.33257605, 1.60575909, 0.07567478,
       0.49312409, 4.38858068, 0.25046998, 0.33576507, 1.60642604,
       0.04462897, 0.07406024, 2.34492424, 0.02327491, 0.08236348,
       3.15179629, 0.04260023, 0.29658136, 4.34981578, 0.04779017,
       0.07615534, 2.29392249, 0.02308668, 0.11837943, 3.62158332,
       0.21582755, 0.79248427, 3.07973482, 0.03437881, 0.23292988,
       4.21327229, 0.0367683 , 0.25229221, 4.26385773, 0.02362778,
       0.08022329, 3.1067407 , 0.04701862, 0.07562543, 2.3058258 ,
       0.25526582, 0.77757305, 2.66894124, 0.11495152, 0.6448419 ,
       4.1013547 , 0.02891942, 0.07008007, 2.75787135, 0.11762039,
       0.14293964, 1.7894457 , 0.02955273, 0.06977047, 2.73119953,
       0.02737315, 0.16999218, 3.97178394, 0.18349982, 0.2245382 ,
       1.63835204, 0.2331317 , 0.79101035, 2.89992619, 0.0947036 ,
       0.57551335, 4.27023358, 0.22380744, 0.28631999, 1.60533782,
       0.05221045, 0.3622101 , 4.4171874 , 0.23079903, 0.79165148,
       2.92416738, 0.04271352, 0.29740503, 4.35107196, 0.0825531 ,
       0.10636115, 1.95676028, 0.05263095, 0.07970299, 2.22600683,
       0.21237091, 0.79188417, 3.11567415, 0.04955153, 0.34486452,
       4.40541405, 0.21214288, 0.79183423, 3.11804551, 0.08335692,
       0.10714706, 1.95169087, 0.05027143, 0.34961877, 4.40904436,
       0.07690942, 0.10092378, 1.99466369, 0.0277318 , 0.17354991,
       3.98950563, 0.2010071 , 0.78784038, 3.23391347, 0.02410996,
       0.13359709, 3.7475974 , 0.11722243, 0.14249965, 1.79087097,
       0.08349531, 0.10728264, 1.9508258 , 0.02789525, 0.1751535 ,
       3.9972963 , 0.2949088 , 0.70733939, 2.22614015, 0.29240062,
       0.71472059, 2.25753556, 0.02586692, 0.15433193, 3.88587293,
       0.16068575, 0.74724699, 3.65237361, 0.30434186, 0.67101534,
       2.09419035, 0.04655359, 0.3245849 , 4.38635507, 0.02274987,
       0.1117191 , 3.557433  , 0.05073103, 0.35263126, 4.41118538,
       0.13794345, 0.70470706, 3.88244818, 0.02657247, 0.16183099,
       3.92871108, 0.08760346, 0.54691584, 4.32046236, 0.05960621,
       0.40754045, 4.4298336 , 0.22721804, 0.29217034, 1.60436686,
       0.1013102 , 0.60003431, 4.21868645, 0.07615736, 0.49545031,
       4.38631635, 0.02503119, 0.07512391, 2.9765495 , 0.26319657,
       0.76918354, 2.58510811, 0.2107881 , 0.79151119, 3.13213514,
       0.04115215, 0.07203325, 2.40889596, 0.03078188, 0.20205268,
       4.11193665, 0.28836182, 0.72533279, 2.30628024, 0.15458705,
       0.18639996, 1.68711581, 0.30777202, 0.65135073, 2.03461428])


# switch to the multicellular stochastic model
from local_models.stoch_model_final import param, GonzeModelManyCells

# note that these relative strengths only are about IN THE ABSENCE OF THE OTHER

# make the stochastic figure
model = GonzeModelManyCells(param, initial_values=y0_random)
wt_trajectories = model.run(show_labels=False, seed=0)
wt_ts = wt_trajectories[0][:,0]
wt_avpsol = wt_trajectories[0][:,1:(AVPcells*4+1)]
wt_vipsol = wt_trajectories[0][:,(AVPcells*4+1):(AVPcells*4+VIPcells*4+1)]
wt_navsol = wt_trajectories[0][:,(AVPcells*4+VIPcells*4+1):]

# avp bmalko
avp_model = GonzeModelManyCells(param, initial_values=y0_random, bmalko='AVP')
avp_trajectories = avp_model.run(show_labels=False, seed=0)
avp_ts = avp_trajectories[0][:,0]
avp_avpsol = avp_trajectories[0][:,1:(AVPcells*4+1)]
avp_vipsol = avp_trajectories[0][:,(AVPcells*4+1):(AVPcells*4+VIPcells*4+1)]
avp_navsol = avp_trajectories[0][:,(320+1):]

vip_model = GonzeModelManyCells(param, initial_values=y0_random, bmalko='VIP')
vip_trajectories = vip_model.run(show_labels=False, seed=0)
vip_ts = vip_trajectories[0][:,0]
vip_avpsol = vip_trajectories[0][:,1:(AVPcells*4+1)]
vip_vipsol = vip_trajectories[0][:,(AVPcells*4+1):(AVPcells*4+VIPcells*4+1)]
vip_navsol = vip_trajectories[0][:,(AVPcells*4+VIPcells*4+1):]


# figure
plo.PlotOptions(ticks='in')
plt.figure(figsize=(3.5,3))
gs = gridspec.GridSpec(3,1)

ax = plt.subplot(gs[0,0])

ax.plot(wt_ts/wt_T, wt_navsol[:,::3], 'goldenrod', alpha=0.04, ls='-.')
ax.plot(wt_ts/wt_T, wt_vipsol[:,::4], 'darkorange', alpha=0.13, ls='--')
ax.plot(wt_ts/wt_T, wt_avpsol[:,::4], 'green', alpha=0.07)
ax.plot(wt_ts/wt_T, wt_vipsol[:,::4].mean(1), 'darkorange', alpha=1, ls='--')
ax.plot(wt_ts/wt_T, wt_navsol[:,::3].mean(1), 'goldenrod', alpha=1, ls='-.')
ax.plot(wt_ts/wt_T, wt_avpsol[:,::4].mean(1), 'green', alpha=1)

ax.set_xticks([0,1,2,3,4,5,6])
ax.set_xlim([0,6])
ax.set_xticklabels([])
ax.legend()
ax.set_ylim([0,300])
ax.set_yticks([0,100,200,300])

bx = plt.subplot(gs[1,0])

bx.plot(avp_ts/wt_T, avp_navsol[:,::3], 'goldenrod', alpha=0.04, ls='-.')
bx.plot(avp_ts/wt_T, avp_avpsol[:,::4], 'green', alpha=0.07)
bx.plot(avp_ts/wt_T, avp_vipsol[:,::4], 'darkorange', alpha=0.13, ls='--')
bx.plot(avp_ts/wt_T, avp_avpsol[:,::4].mean(1), 'green', alpha=1)
bx.plot(avp_ts/wt_T, avp_vipsol[:,::4].mean(1), 'darkorange', alpha=1, ls='--')
bx.plot(avp_ts/wt_T, avp_navsol[:,::3].mean(1), 'goldenrod', alpha=1, ls='-.')

bx.set_xticks([0,1,2,3,4,5,6])
bx.set_xticklabels([])
bx.set_xlim([0,6])
bx.set_ylim([0,300])
bx.set_yticks([0,100,200,300])


cx = plt.subplot(gs[2,0])
cx.plot(vip_ts/wt_T, vip_navsol[:,::3], 'goldenrod', alpha=0.04, ls='-.')
cx.plot(vip_ts/wt_T, vip_vipsol[:,::4], 'darkorange', alpha=0.13, ls='--')
cx.plot(vip_ts/wt_T, vip_avpsol[:,::4], 'green', alpha=0.07)
cx.plot(vip_ts/wt_T, vip_vipsol[:,::4].mean(1), 'darkorange', alpha=1, ls='--')
cx.plot(vip_ts/wt_T, vip_navsol[:,::3].mean(1), 'goldenrod', alpha=1, ls='-.')
cx.plot(vip_ts/wt_T, vip_avpsol[:,::4].mean(1), 'green', alpha=1)

cx.set_xticks([0,1,2,3,4,5,6])
cx.set_xlim([0,6])
cx.set_ylim([0,300])
cx.set_yticks([0,100,200,300])
cx.set_xlabel('Day')

plt.legend()
plt.tight_layout(**plo.layout_pad)
plt.savefig('results/model_figure_data.pdf')
plt.show()


# export data
header = ['TimesH', 'Frame']+['AVP'+str(i) for i in range(AVPcells)]+['VIP'+str(i) for i in range(VIPcells)]+['NAV'+str(i) for i in range(NAVcells)]

# wt traces
output_data = np.hstack([np.array([wt_ts*24/wt_T]).T,  np.array([np.arange(len(wt_ts))]).T, wt_avpsol[:,::4], wt_vipsol[:,::4], wt_navsol[:,::3] ])
output_df = pd.DataFrame(data=output_data,
            columns = header)
output_df.to_csv('results/WT_finalmodel_trajectories.csv', index=False)

# avp traces
output_data = np.hstack([np.array([avp_ts*24/wt_T]).T, np.array([np.arange(len(avp_ts))]).T, avp_avpsol[:,::4], avp_vipsol[:,::4], avp_navsol[:,::3] ])
output_df = pd.DataFrame(data=output_data,
            columns = header)
output_df.to_csv('results/AVPBmalKO_finalmodel_trajectories.csv', index=False)

# vip traces
output_data = np.hstack([np.array([vip_ts*24/wt_T]).T, np.array([np.arange(len(vip_ts))]).T, vip_avpsol[:,::4], vip_vipsol[:,::4], vip_navsol[:,::3] ])
output_df = pd.DataFrame(data=output_data,
            columns = header)
output_df.to_csv('results/VIPBmalKO_finalmodel_trajectories.csv', index=False)




##############################################################################
# perform simulations and statistical tests
##############################################################################

# try to load the simulation
try:    
    with open("data/wt_final.pickle", "rb") as read_file:
        wt_trajectories = pickle.load(read_file)
    with open("data/avp_final.pickle", "rb") as read_file:
        avp_trajectories = pickle.load(read_file)
    with open("data/vip_final.pickle", "rb") as read_file:
        vip_trajectories = pickle.load(read_file)
    print "Loaded final simulation."
    traj = {'wt': wt_trajectories,
              'avp': avp_trajectories,
              'vip': vip_trajectories}
except IOError:
    print "Final simulation does not exist yet."
    print "Simulating 5 iterations of final model, as in experiment."
    wt_trajectories = []
    avp_trajectories = []
    vip_trajectories = []
    for tn in range(5):
        print tn,
        # get random initial condition
        # initial phases
        init_conditions_AV  = [single_osc.lc(wt_T*np.random.rand()) 
                                for i in range(AVPcells+VIPcells)]
        init_conditions_NAV = [single_osc.lc(wt_T*np.random.rand())[:-1]
                                for i in range(NAVcells)]
        y0_random = np.hstack(init_conditions_AV+init_conditions_NAV)

        # do the simulation
        model = GonzeModelManyCells(param, initial_values=y0_random)
        wt_trajectories.append(model.run(show_labels=False, seed=0))

        # avp bmalko
        avp_model = GonzeModelManyCells(param, bmalko='AVP', 
                                        initial_values=y0_random)
        avp_trajectories.append(avp_model.run(show_labels=False, seed=0))

        # vip bmalko
        vip_model = GonzeModelManyCells(param, bmalko='VIP', 
                                        initial_values=y0_random)
        vip_trajectories.append(vip_model.run(show_labels=False, seed=0))

    # save results
    with open("data/wt_final.pickle", "wb") as output_file:
            pickle.dump(wt_trajectories, output_file)
    with open("data/avp_final.pickle", "wb") as output_file:
            pickle.dump(avp_trajectories, output_file)
    with open("data//vip_final.pickle", "wb") as output_file:
            pickle.dump(vip_trajectories, output_file)

    traj = {'wt': wt_trajectories,
              'avp': avp_trajectories,
              'vip': vip_trajectories}


# MIC Calculations
try:
    # load the results
    with open("results/finalmodel_mic_results.pickle", "rb") as input_file:
        results = pickle.load(input_file)
except IOError:
    print "MIC has not been calculated yet."
    print "Performing MIC calcualtion."
    # now perform MIC calculation
    def mic_of_simulation(trajectories):
        """
        returns the MIC values for one set of the SCN trajectories in question
        """

        avpvipsol = trajectories[:, 1:(VIPcells*4+AVPcells*4+1)]
        navsol = trajectories[:, (VIPcells*4+AVPcells*4+1):]

        per2 = np.hstack([avpvipsol[:, ::4], navsol[:, ::3]])
        numcells = per2.shape[1]

        # set up mic calculator
        mic = mp.MINE(alpha=0.6, c=15, est='mic_approx')
        mic_values = []
        for combo in combinations(range(numcells), 2):
            mic.compute_score(per2[:, combo[0]], per2[:, combo[1]])
            mic_values.append(mic.mic())

        return mic_values

    # process wt
    print "WT"
    wt_traj = traj['wt']
    wt = []
    for idx, ti in enumerate(wt_traj):
        print idx,
        mic_mean = np.mean(mic_of_simulation(ti[0]))
        wt.append(mic_mean)

    # process avp
    print "AVP-BmalKO"
    avp_traj = traj['avp']
    avp = []
    for idx, ti in enumerate(avp_traj):
        print idx,
        mic_mean = np.mean(mic_of_simulation(ti[0]))
        avp.append(mic_mean)

    # process vip
    print "VIP-BmalKO"
    vip_traj = traj['vip']
    vip = []
    for idx, ti in enumerate(vip_traj):
        print idx,
        mic_mean = np.mean(mic_of_simulation(ti[0]))
        vip.append(mic_mean)

    results = [wt, avp, vip]

    with open("results/finalmodel_mic_results.pickle", "wb") as output_file:
        pickle.dump(results, output_file)



# run the stats
# compare avpbmalko and vipbmalko vs. wt for all cases
# running a mann-whitney U test for nonparametric comparison
# recap of phenotypes is: AVPBmal1ko < VIPBmal1KO = WT
# signifiance of p < 0.05
# correct p-values with Bonferroni correction
# at each level, we are comparing: WT:AVP, WT:VIP, VIP:AVP so 3*7=21
# comparisons

r = results
wa = stats.mannwhitneyu(r[0], r[1], alternative='two-sided')[1]
wv = stats.mannwhitneyu(r[0], r[2], alternative='two-sided')[1]
av = stats.mannwhitneyu(r[1], r[2], alternative='two-sided')[1]
u_results = [wa, wv, av]
print u_results
ur = u_results
correct_phenotype1 = np.all([ur[0]<0.05, ur[1]>0.05, ur[2]<0.05]) 







