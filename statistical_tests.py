"""

Uses scipy to perform a mann-whitney u test on the results of the MIC 
calculation on the simulated SCNs.

author: John Abel
"""
from __future__ import division
import pickle
import sys
assert sys.version[0] == '2', "This file is designed for python2"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats

from local_imports import PlotOptions as plo

# first, the figure for parameter switching
kavs = [0.1, 0.2, 0.5, 1, 2, 5, 10]

# load the results
with open("results/params_mic_results.pickle", "rb") as input_file:
    results = pickle.load(input_file)

# run the stats
# compare avpbmalko and vipbmalko vs. wt for all cases
# running a mann-whitney U test for nonparametric comparison
# recap of phenotypes is: AVPBmal1ko < VIPBmal1KO = WT
# signifiance of p < 0.05
# correct p-values with Bonferroni correction
# at each level, we are comparing: WT:AVP, WT:VIP, VIP:AVP so 3*7=21
# comparisons
u_results = []
for r in results:
    wa = stats.mannwhitneyu(r[0], r[1], alternative='two-sided')[1]
    wv = stats.mannwhitneyu(r[0], r[2], alternative='two-sided')[1]
    av = stats.mannwhitneyu(r[1], r[2], alternative='two-sided')[1]
    u_results.append([wa, wv, av])
u_results = np.array(u_results)*21

correct_phenotype = [np.all([ur[0]<0.05, ur[1]>0.05, ur[2]<0.05]) 
                        for ur in u_results]

# plot the figure

plo.PlotOptions(ticks='in')
plt.figure()
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0, 0])

for idx, kav in enumerate(kavs):
    l1 = ax.plot(plo.jitter_uni([kav]*100, x_range=kav/10),
                 results[idx][0], 'o', c='k', alpha=0.12)

for idx, kav in enumerate(kavs):
    l2 = ax.plot(plo.jitter_uni([kav]*100, x_range=kav/10),
                 results[idx][1], 'o', c='j', alpha=0.12)

for idx, kav in enumerate(kavs):
    l3 = ax.plot(plo.jitter_uni([kav]*100, x_range=kav/10),
                 results[idx][2], 'fo', alpha=0.12)

ax.plot(kavs, [np.mean(r[0]) for r in results], 'k', label='WT Mean')
ax.plot(kavs, [np.mean(r[1]) for r in results], 'j', label='AVPBmal1KO Mean')
ax.plot(kavs, [np.mean(r[2]) for r in results], 'f', label='VIPBmal1KO Mean')
ax.fill_between([4,20],[0.7,0.7], color='0.8')

ax.set_xscale('log')
ax.set_xlabel('Coupling Strength AVP/VIP')
ax.set_ylabel('Resynchrony MIC')
ax.legend()

ax.set_ylim([0.2, 0.7])
ax.set_xlim([0.07, 18])
plt.tight_layout(**plo.layout_pad)

plt.savefig('results/params_stats_fig.svg')
plt.show()





# next, the figure for varyign cell type fractions
navps = np.array([4, 7, 13, 20, 27, 33, 36])
nvips = 40-navps
type_ratios = navps/nvips

# load the results
with open("results/celltypes_mic_results.pickle", "rb") as input_file:
    results = pickle.load(input_file)

# run the stats
# compare avpbmalko and vipbmalko vs. wt for all cases
# running a mann-whitney U test for nonparametric comparison
# recap of phenotypes is: AVPBmal1ko < VIPBmal1KO = WT
# signifiance of p < 0.05
# correct p-values with Bonferroni correction
# at each level, we are comparing: WT:AVP, WT:VIP, VIP:AVP so 3*7=21
# comparisons
u_results = []
for r in results:
    wa = stats.mannwhitneyu(r[0], r[1], alternative='two-sided')[1]
    wv = stats.mannwhitneyu(r[0], r[2], alternative='two-sided')[1]
    av = stats.mannwhitneyu(r[1], r[2], alternative='two-sided')[1]
    u_results.append([wa, wv, av])
u_results = np.array(u_results)*21

correct_phenotype = [np.all([ur[0]<0.05, ur[1]>0.05, ur[2]<0.05]) 
                        for ur in u_results]

# plot the figure

plo.PlotOptions(ticks='in')
plt.figure()
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0, 0])

for idx, kav in enumerate(kavs):
    l1 = ax.plot(plo.jitter_uni([kav]*100, x_range=kav/10),
                 results[idx][0], 'o', c='k', alpha=0.12)

for idx, kav in enumerate(kavs):
    l2 = ax.plot(plo.jitter_uni([kav]*100, x_range=kav/10),
                 results[idx][1], 'o', c='j', alpha=0.12)

for idx, kav in enumerate(kavs):
    l3 = ax.plot(plo.jitter_uni([kav]*100, x_range=kav/10),
                 results[idx][2], 'fo', alpha=0.12)

ax.plot(type_ratios, [np.mean(r[0]) for r in results], 'k', label='WT Mean')
ax.plot(type_ratios, [np.mean(r[1]) for r in results], 'j', 
    label='AVPBmal1KO Mean')
ax.plot(type_ratios, [np.mean(r[2]) for r in results], 'f', 
    label='VIPBmal1KO Mean')
ax.fill_between([7.5,20],[0.7,0.7], color='0.8')

ax.set_xscale('log')
ax.set_xlabel('Cell Type Ratio AVP/VIP')
ax.set_ylabel('Resynchrony MIC')
ax.legend()

ax.set_ylim([0.2, 0.7])
ax.set_xlim([0.07, 18])
plt.tight_layout(**plo.layout_pad)

plt.savefig('results/celltypes_stats_fig.svg')
plt.show()

