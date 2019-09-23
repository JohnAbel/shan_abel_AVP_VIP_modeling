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

To generate the example trajectories of Fig 6, run `run_final_scn_model.py`.

To perform the parameter sweep for neurotransmission pathway strength, run 
`simulate_changing_parameter.py`, and to perform the parameter sweep for
ratios of cell types, run `simulate_changing_celltypes.py`.

Once these simulationa are run, use `perform_mic_calculation.py` for
calculating MIC, then `stats_changing_parameter.py` for the statistical 
analysis and Fig 5.

For questions, contact abelj at mit dot edu, or jhabel01 at gmail dot com.
