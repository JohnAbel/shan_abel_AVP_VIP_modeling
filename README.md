# Modeling Code for Shan, Abel ... Doyle III, Takahashi

This code performs all deterministic and stochastic simulations from 
Shan et al., currently in review. This code was written in Python 2.7. A 
full list of dependencies are as follows:

| Package | Version |
|------------|---------|
| Python | 2.7.3 |
| numpy | 1.15.1 |
| scipy | 1.2.1 |
| matplotlib | 2.2.3 |
| CasADi | 2.3.0 |
| gillespy | 1.1 |
| pandas | 0.23.4 |
| minepy | 1.2.2 |

To generate the example trajectories of Fig 6, run `example_5x_scn.py`.
To perform the parameter sweep for relative coupling strength, run 
`simulate_changing_coupling.py`, followed by `perform_mic_calculation.py` for
calculating MIC, then `stats_changing_parameter.py` for the statistical 
analysis and Fig S9C.

For questions, contact jhabel01 at gmail dot com.
