"""
This file performs 100 simulations of WT, AVPBmalKO, VIPBmalKO for varying
the AVP:VIP coupling strength ratios from 0.1 to 10. Files are saved in folder
"data" for next step of processing.

John Abel
"""

from __future__ import division
import pickle

import numpy as np

from local_imports import LimitCycle as lc
from local_models import gonze_model as ab
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
from local_models.gonze_stoch_multi_celltypes import param, GonzeModelManyCells

def load_trajectories(navp, nvip):
    """
    Loads simulated trajectories
    """
    # try to load the simulation
    try:    
        with open("data/celltypes/wt_"+str(navp)+
                  "_"+str(nvip)+".pickle", "rb") as read_file:
            wt_trajectories = pickle.load(read_file)
        with open("data/celltypes/avp_"+str(navp)+
                  "_"+str(nvip)+".pickle", "rb") as read_file:
            avp_trajectories = pickle.load(read_file)
        with open("data/celltypes/vip_"+str(navp)+
                  "_"+str(nvip)+".pickle", "rb") as read_file:
            vip_trajectories = pickle.load(read_file)
        print "Loaded "+str(navp)+str(nvip)

    except IOError:
        print str(navp)+str(nvip)+" does not exist yet."

    return {'wt': wt_trajectories,
            'avp': avp_trajectories,
            'vip': vip_trajectories}

def simulate_trajectories(navp):
    """
    Simulates and saves desired trajectories.
    """
    nvip = 40-navp
    print "Simulating "+str(navp)+" "+str(nvip)
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
        model = GonzeModelManyCells(param, AVPcells=navp,
                VIPcells=nvip, initial_values=y0_random)
        wt_trajectories.append(model.run(show_labels=False, seed=0))

        # avp bmalko
        avp_model = GonzeModelManyCells(param, bmalko='AVP', AVPcells=navp,
                VIPcells=nvip, initial_values=y0_random)
        avp_trajectories.append(avp_model.run(show_labels=False, seed=0))

        # vip bmalko
        vip_model = GonzeModelManyCells(param, bmalko='VIP', AVPcells=navp,
                VIPcells=nvip, initial_values=y0_random)
        vip_trajectories.append(vip_model.run(show_labels=False, seed=0))

    # save results
    with open("data/celltypes/wt_"+str(navp)+
                  "_"+str(nvip)+".pickle", "wb") as output_file:
            pickle.dump(wt_trajectories, output_file)
    with open("data/celltypes/avp_"+str(navp)+
                  "_"+str(nvip)+".pickle", "wb") as output_file:
            pickle.dump(avp_trajectories, output_file)
    with open("data/celltypes/vip_"+str(navp)+
                  "_"+str(nvip)+".pickle", "wb") as output_file:
            pickle.dump(vip_trajectories, output_file)

    return {'wt': wt_trajectories,
            'avp': avp_trajectories,
            'vip': vip_trajectories}

navps = [4, 7, 13, 20, 27, 33, 36]

for navp in navps:
    simulate_trajectories(navp)
    traj = load_trajectories(navp, 40-navp)


print "All trajectories simulated successfully."
