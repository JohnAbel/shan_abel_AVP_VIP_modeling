"""
Created on Tue Jan 13 13:01:35 2014

@author: John H. Abel

This is also the first model build using native gillespy, so let's see how
it goes. - jha
"""

# common imports
from __future__ import division
from collections import OrderedDict

# python packages
import numpy as np
import casadi as cs
import gillespy as gsp

modelversion = 'gonze_model'

# constants and equations setup, trying a new method
EqCount = 40+40+30
ParamCount  = 20

param = [  0.7,    1,    4, 0.35,    1,  0.7, 0.35,   
             1,  0.7, 0.35,    1, 0.35,    1,    1,
           0.4,    1,  0.75,    0 
           ]
y0in = np.ones(EqCount)
period = 30.27

def ODEmodel(ko=None, bmalko=None, kav=5):

    AVPcells = 20
    VIPcells = 20
    NAVcells = 20

    state_dict = OrderedDict()
    for cellidx in range(AVPcells):
        # first compartment: AVP
        state_dict['X1'+str(cellidx)]    = cs.SX.sym("X1"+str(cellidx))
        state_dict['Y1'+str(cellidx)]    = cs.SX.sym("Y1"+str(cellidx))
        state_dict['Z1'+str(cellidx)]    = cs.SX.sym("Z1"+str(cellidx))
        state_dict['A1'+str(cellidx)]    = cs.SX.sym("A1"+str(cellidx))

    for cellidx in range(VIPcells):
        # second compartment: VIP
        state_dict['X2'+str(cellidx)]    = cs.SX.sym("X2"+str(cellidx))
        state_dict['Y2'+str(cellidx)]    = cs.SX.sym("Y2"+str(cellidx))
        state_dict['Z2'+str(cellidx)]    = cs.SX.sym("Z2"+str(cellidx))
        state_dict['V2'+str(cellidx)]    = cs.SX.sym("V2"+str(cellidx))

    for cellidx in range(NAVcells):
        # second compartment: VIP
        state_dict['X3'+str(cellidx)]    = cs.SX.sym("X3"+str(cellidx))
        state_dict['Y3'+str(cellidx)]    = cs.SX.sym("Y3"+str(cellidx))
        state_dict['Z3'+str(cellidx)]    = cs.SX.sym("Z3"+str(cellidx))
    
    sd = state_dict

    #for Casadi
    y = cs.vertcat(sd.values())
    t = cs.SX.sym("t")

    # Parameter Assignments
    v1  = cs.SX.sym('v1')
    K1  = cs.SX.sym('K1')
    n   = cs.SX.sym('n')
    v2  = cs.SX.sym('v2')
    K2  = cs.SX.sym('K2')
    k3  = cs.SX.sym('k3')
    v4  = cs.SX.sym('v4')
    K4  = cs.SX.sym('K4')
    k5  = cs.SX.sym('k5')
    v6  = cs.SX.sym('v6')
    K6  = cs.SX.sym('K6')
    k7  = cs.SX.sym('k7')
    v8  = cs.SX.sym('v8')
    K8  = cs.SX.sym('K8')
    vc  = cs.SX.sym('vc')
    Kc  = cs.SX.sym('Kc')
    K   = cs.SX.sym('K')
    L   = cs.SX.sym('L')

    
    param_set = cs.vertcat([v1,K1,n,v2,K2,k3,v4,K4,k5,v6,K6,k7,v8,K8,vc,Kc,K,L])

    # ratio of AVP to VIP strengths
    ar = kav/(kav+1)
    vr = 1/(kav+1)

    # set up kos, bmalkos
    vipko=1
    avpko=1
    avpbmalko = 1
    vipbmalko = 1
    bmalko_v1 = 0.

    # set up kos
    if ko=='VIP':
        vipko=0
    if ko=='AVP':
        avpko=0
    if ko=='AVPVIP':
        vipko=0
        avpko=0
    if bmalko=='AVP':
        avpbmalko = bmalko_v1
    elif bmalko=='VIP':
        vipbmalko = bmalko_v1
    elif bmalko=='AVPVIP':
        vipbmalko=bmalko_v1
        avpbmalko=bmalko_v1

    ode = [[]]*(4*AVPcells+4*VIPcells+3*NAVcells)

    coupling = ar*(sd['A10']+sd['A11']+sd['A12']+sd['A13']+sd['A14']+sd['A15']+sd['A16']+sd['A17']+sd['A18']+sd['A19']+sd['A110']+sd['A111']+sd['A112']+sd['A113']+sd['A114']+sd['A115']+sd['A116']+sd['A117']+sd['A118']+sd['A119'])/20 +\
            vr*(sd['V20']+sd['V21']+sd['V22']+sd['V23']+sd['V24']+sd['V25']+sd['V26']+sd['V27']+sd['V28']+sd['V29']+sd['V210']+sd['V211']+sd['V212']+sd['V213']+sd['V214']+sd['V215']+sd['V216']+sd['V217']+sd['V218']+sd['V219'])/20

    # p3 is assigned manually to vary period
    if p3 is None:
        p3 = [v2]*(AVPcells+VIPcells+NAVcells)

    for cellidx in range(AVPcells):

        # start where there is no data yet
        start_offset = 4*cellidx
        ci = str(cellidx)
        p3idx = cellidx

        ode[0+start_offset] = avpbmalko*v1*K1**n/(K1**n + sd['Z1'+ci]**n) \
                 - p3[p3idx]*(sd['X1'+ci])/(K2+sd['X1'+ci]) \
                 +vc*K*(coupling)/(Kc +K*coupling)
                 
        ode[1+start_offset] = k3*(sd['X1'+ci]) - v4*sd['Y1'+ci]/(K4+sd['Y1'+ci])
        
        ode[2+start_offset] = k5*sd['Y1'+ci] - v6*sd['Z1'+ci]/(K6+sd['Z1'+ci])
        ode[3+start_offset] = avpbmalko*avpko*k7*(sd['X1'+ci]) - v8*sd['A1'+ci]/(K8+sd['A1'+ci])

    for cellidx in range(VIPcells):

        # start where there is no data yet
        start_offset = 4*cellidx+4*AVPcells
        ci = str(cellidx)
    
        ode[0+start_offset] = vipbmalko*v1*K1**n/(K1**n + sd['Z2'+ci]**n) \
                 - p3[p3idx]*(sd['X2'+ci])/(K2+sd['X2'+ci]) \
                 +vc*K*coupling/(Kc +K*coupling)
                 
        ode[1+start_offset] = k3*(sd['X2'+ci]) - v4*sd['Y2'+ci]/(K4+sd['Y2'+ci])
        
        ode[2+start_offset] = k5*sd['Y2'+ci] - v6*sd['Z2'+ci]/(K6+sd['Z2'+ci])
        ode[3+start_offset] = vipko*k7*(sd['X2'+ci]) - v8*sd['V2'+ci]/(K8+sd['V2'+ci])

    for cellidx in range(NAVcells):

        # start where there is no data yet
        start_offset = 3*cellidx+4*AVPcells+4*VIPcells
        ci = str(cellidx)
    
        ode[0+start_offset] = v1*K1**n/(K1**n + sd['Z3'+ci]**n) \
                 - p3[p3idx]*(sd['X3'+ci])/(K2+sd['X3'+ci]) \
                 +vc*K*(coupling)/(Kc +K*coupling)
                 
        ode[1+start_offset] = k3*(sd['X3'+ci]) - v4*sd['Y3'+ci]/(K4+sd['Y3'+ci])
        
        ode[2+start_offset] = k5*sd['Y3'+ci] - v6*sd['Z3'+ci]/(K6+sd['Z3'+ci])


    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t,x=y,p=param_set), 
            cs.daeOut(ode=ode))

    fn.setOption("name","gonze_model")

    return fn