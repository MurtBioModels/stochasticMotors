from motorgillespie.plotting import indlude_in_report as ir
from motorgillespie.analysis import print_info as pi
'''Analysis of motor objects obtained from script external_force_init.py'''

## Simulation settings ##
dirct1 = '20230414_125004_external_force_100_False_allbound'
tslist = [1, 2, 3, 4]
kmlist = [0.15, 0.1, 0.25, 0.2, 0.35, 0.3, 0.4] # HAS TO BE THE RIGHT ORDER!!!!
fexlist = [-1, -2, -3, -4, -5, -6, 0] # HAS TO BE THE RIGHT ORDER!!!!

#pi.inspect(dirct=dirct1)

## CARGO ##
# Run Length
#ir.rl_bead_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, filename='')
#ir.plot_n_fex_km_rl(dirct=dirct1, filename='rl_cargo_N_fex_km_csv', n_include=tslist, fex_include=fexlist, km_include=kmlist, show=False)

# Bind time
#ir.bt_bead_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, filename='')
#ir.plot_n_fex_km_bt(dirct=dirct1, filename='bt_N_fex_km_.csv', n_include=tslist, fex_include=fexlist, km_include=kmlist, show=False, figname='')

# velocity
#ir.vel_bead_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, filename='')
#ir.plot_n_fex_km_vel(dirct=dirct1, filename='vel_N_fex_km_.csv', n_include=tslist, fex_include=fexlist, km_include=kmlist, show=False, figname='')

# xb
#ir.xb_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.1, filename='')
ir.plot_n_fex_km_xb(dirct=dirct1, filename='xb_sampled_.csv', n_include=tslist, fex_include=fexlist, km_include=kmlist, stat='probability', show=False, figname='')

# trajectories
ir.traj_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, show=False)

# Bound motors
ir.bound_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.01, filename='')
ir.plot_n_fex_km_boundmotors(dirct=dirct1, filename='', n_include=tslist, fex_include=fexlist, km_include=kmlist, show=True, figname='')
# Motor unbinding events
ir.unbindevent_bead_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, filename='')
ir.plot_n_fex_km_unbindevent(dirct=dirct1, filename='', n_include=tslist, fex_include=fexlist, km_include=kmlist, show=True, figname='')
# Segments
#ir.segment_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, filename='')
#ir.plot_n_fex_km_seg(dirct=dirct1, filename='segments_.csv', n_include=tslist, fex_include=fexlist, km_include=kmlist, show=False, figname='')
#pi.inspect(dirct=dirct1)


# motor forces
ir.motorforces_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.1, samplesize=100, filename='')
ir.plot_fex_N_km_forces_motors(dirct=dirct1, filename='', n_include=tslist, fex_include=fexlist, km_include=kmlist, stat='probability', show=True, figname='')
# motor displacement
ir.xm_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, stepsize=0.01, samplesize=100, filename='')
ir.plot_fex_N_km_xm(dirct=dirct1, filename='', n_include=tslist, fex_include=fexlist, km_include=kmlist, stat='probability', show=True, figname='')
# motor runlengths
ir.rl_motors_n_fex_km(dirct=dirct1, ts_list=tslist, fex_list=fexlist, km_list=kmlist, filename='')
ir.plot_fex_N_km_rl_motors(dirct=dirct1, filename='', n_include=tslist, fex_include=fexlist, km_include=kmlist, show=True, figname='')
