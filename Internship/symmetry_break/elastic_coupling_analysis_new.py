from motorgillespie.plotting import indlude_in_report as ir

'''Analysis of motor objects obtained from script elastic_coupling_init_new.py'''

## Simulation settings ##
dirct1 = '20230414_125004_elasticcoupling_100_False_allbound'
tslist = [1, 2, 3, 4]
kmlist = [0.15, 0.1, 0.25, 0.2, 0.35, 0.3, 0.4] # HAS TO BE THE RIGHT ORDER!!!!
fexlist = [-1, -2, -3, -4, -5, -6, 0] # HAS TO BE THE RIGHT ORDER!!!!

## CARGO ##
ir.rl_bead_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, filename='')
ir.plot_n_fex_km_rl(dirct=dirct1, filename='', km_include=(0.15, 0.1, 0.25, 0.2, 0.35, 0.3, 0.4), show=True)


