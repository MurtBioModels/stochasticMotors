from motorgillespie.plotting import indlude_in_report as ir
from motorgillespie.analysis import print_info as pi
'''Analysis of motor objects obtained from script external_force_init.py'''

## Simulation settings ##
dirct1 = '20230414_125004_external_force_100_False_allbound'
tslist = [1, 2, 3, 4]
kmlist = [0.15, 0.1, 0.25, 0.2, 0.35, 0.3, 0.4] # HAS TO BE THE RIGHT ORDER!!!!
fexlist = [-1, -2, -3, -4, -5, -6, 0] # HAS TO BE THE RIGHT ORDER!!!!

## CARGO ##
# Run Length
#ir.rl_bead_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, filename='')
#ir.plot_n_fex_km_rl(dirct=dirct1, filename='rl_cargo_N_fex_km_csv', km_include=(0.15, 0.1, 0.25, 0.2, 0.35, 0.3, 0.4), show=True)
# Segments
ir.segment_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, filename='')
ir.plot_n_fex_km_seg(dirct=dirct1, filename='segments_.csv', n_include=tslist, fex_include=fexlist, km_include=kmlist, show=False, figname='')
#pi.inspect(dirct=dirct1)
