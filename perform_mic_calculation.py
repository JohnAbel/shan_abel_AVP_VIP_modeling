"""
Run model of the full SCN consisting of 60 Gonze oscillators. The period
is found using the deterministic model to normalize the resulting trajectories.
The initial conditions are random--these were hard-coded in to return an identical 
figure every time, but they can be commented out for random start.


Generate the final figure, save the trajectories.

John Abel
"""

from __future__ import division
import sys
assert sys.version[0]=='2', "This file must be run in python 2"
import pickle
from multiprocessing import Pool


def oneshot_mics(kav):
    """
    Incorporates elements of all three functions above in order to
    process all the trajectories in one place. The imports exist so this
    can be done in parallel. kav, as before is the ratio of avp to vip signal 
    strength.
    """
    from itertools import combinations
    import numpy as np
    import minepy as mp
    import pickle 

    def mic_of_simulation(trajectories):
        """
        returns the MIC values for one set of the SCN trajectories in question
        """

        avpsol = trajectories[:, 1:(80+1)]
        vipsol = trajectories[:, (80+1):(160+1)]
        navsol = trajectories[:, (160+1):]

        per2 = np.hstack([avpsol[:, ::4], vipsol[:, ::4], navsol[:, ::3]])
        numcells = per2.shape[1]

        # set up mic calculator
        mic = mp.MINE(alpha=0.6, c=15, est='mic_approx')
        mic_values = []
        for combo in combinations(range(numcells), 2):
            mic.compute_score(per2[:, combo[0]], per2[:, combo[1]])
            mic_values.append(mic.mic())

        return mic_values


    # get the dict
    try:    
        with open("Data/wt_"+str(kav)+".pickle", "rb") as read_file:
            wt_trajectories = pickle.load(read_file)
        with open("Data/avp_"+str(kav)+".pickle", "rb") as read_file:
            avp_trajectories = pickle.load(read_file)
        with open("Data/vip_"+str(kav)+".pickle", "rb") as read_file:
            vip_trajectories = pickle.load(read_file)
        print "Loaded "+str(kav)

    except IOError:
        print str(kav)+" does not exist yet."

    traj = {'kav': kav,
            'wt' : wt_trajectories,
            'avp': avp_trajectories,
            'vip': vip_trajectories}

    # process wt
    wt_traj = traj['wt']
    wt = []
    for idx, ti in enumerate(wt_traj):
        print idx
        mic_mean = np.mean(mic_of_simulation(ti[0]))
        wt.append(mic_mean)

    # process avp
    avp_traj = traj['avp']
    avp = []
    for idx, ti in enumerate(avp_traj):
        print idx
        mic_mean = np.mean(mic_of_simulation(ti[0]))
        avp.append(mic_mean)

    # process vip
    vip_traj = traj['vip']
    vip = []
    for idx, ti in enumerate(vip_traj):
        print idx
        mic_mean = np.mean(mic_of_simulation(ti[0]))
        vip.append(mic_mean)

    return wt, avp, vip

#

# Functions are defined, let's process these MIC scores.

# process these in parallel
kavs = [0.1, 0.2, 0.5, 1, 2, 5, 10]

p = Pool(7)
results = p.map(oneshot_mics, kavs)

with open("Data/mic_results.pickle", "wb") as output_file:
    pickle.dump(results, output_file)

with open("Data/mic_results.pickle", "rb") as input_file:
    results = pickle.load(input_file)

# confirmed saved and opened
