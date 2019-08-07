"""
Run model of the full SCN consisting of 60 Gonze oscillators. The period
is found using the deterministic model to normalize the resulting trajectories.
The initial conditions are random--these were hard-coded in to return an identical 
figure every time, but they can be commented out for random start.


Generate the final figure, save the trajectories.

John Abel
"""

from __future__ import division
import pickle

import numpy as np

from local_imports import LimitCycle as lc
from local_models import gonze_model as ab
reload(ab)
from local_models.gonze_model import param, ODEmodel, EqCount



# find original period
single_osc = lc.Oscillator(ODEmodel(), param, y0=np.ones(EqCount))
ts, sol = single_osc.int_odes(500)
single_osc.approx_y0_T(trans=2000)
wt_T = single_osc.T
single_osc.limit_cycle()

# number of each celltype
AVPcells = 20; VIPcells=20; NAVcells = 20
totcells = AVPcells+VIPcells+NAVcells

# perform sim ulation
# switch to the many cell model
from local_models import gonze_model_manycell as ab
reload(ab)
from local_models.gonze_model_stochastic_manycell import param, GonzeModelManyCells

def load_trajectories(kav):
    """
    Loads simulated trajectories
    """
    # try to load the simulation
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

    return {'wt': wt_trajectories,
            'avp': avp_trajectories,
            'vip': vip_trajectories}

def simulate_trajectories(kav):
    """
    Simulates and saves desired trajectories.
    """
    print "Simulating "+str(kav)
    wt_trajectories = []
    avp_trajectories = []
    vip_trajectories = []
    for tn in range(100):
        # get random initial condition
        # initial phases
        init_conditions_AV  = [single_osc.lc(wt_T*np.random.rand()) 
                                for i in range(AVPcells+VIPcells)]
        init_conditions_NAV = [single_osc.lc(wt_T*np.random.rand())[:-1]
                                for i in range(NAVcells)]
        y0_random = np.hstack(init_conditions_AV+init_conditions_NAV)

        # do the simulation
        model = GonzeModelManyCells(param, kav=kav, 
            initial_values=y0_random)
        wt_trajectories.append(model.run(show_labels=False, seed=0))

        # avp bmalko
        avp_model = GonzeModelManyCells(param, bmalko='AVP', kav=kav, 
            initial_values=y0_random)
        avp_trajectories.append(avp_model.run(show_labels=False, seed=0))

        # vip bmalko
        vip_model = GonzeModelManyCells(param, bmalko='VIP', kav=kav, 
            initial_values=y0_random)
        vip_trajectories.append(vip_model.run(show_labels=False, seed=0))

    # save results
    with open("Data/wt_"+str(kav)+".pickle", "wb") as output_file:
            pickle.dump(wt_trajectories, output_file)
    with open("Data/avp_"+str(kav)+".pickle", "wb") as output_file:
            pickle.dump(avp_trajectories, output_file)
    with open("Data/vip_"+str(kav)+".pickle", "wb") as output_file:
            pickle.dump(vip_trajectories, output_file)

    return {'wt': wt_trajectories,
            'avp': avp_trajectories,
            'vip': vip_trajectories}

kavs = [0.1, 0.2, 0.5, 1, 2, 5, 10]

for kav in kavs:
    simulate_trajectories(kav)
    traj = load_trajectories(kav)


print "All trajectories simulated successfully."