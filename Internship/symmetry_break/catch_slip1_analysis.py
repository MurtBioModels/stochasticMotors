from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
import os

'''Analysis of motor objects obtained from script catch_slip1_init.py'''

## Simulation settings ##
dirct = '20221018_142718_catch_slip1_eps=0.66_equal_km'
team_comb = [(1,0,1,0), (1,1,1,1), (1,2,1,2), (1,3,1,3)] # should be in the right order

# Create lists with
list_fn = []
list_ts = []
for i in team_comb:
    list_fn.append(f'{i}')
    list_ts.append(f'kin,dyn={i}')

for root, subdirs, files in os.walk(f'.\motor_objects\{dirct}'):
    for index, subdir in enumerate(subdirs):
        if subdir == 'figures':
            continue
        bf.xbead_dist(dirct, subdir, f'{list_fn[index]}', list_ts[index], show=False)
        mf.xm_dist(dirct, subdir, f'{list_fn[index]}', list_ts[index], show=False)
        mf.forces_dist(dirct, subdir, f'{list_fn[index]}', list_ts[index], show=False)
        bf.trace_velocity(dirct, subdir, f'{list_fn[index]}', list_ts[index],  stat='count', show=False)
        bf.rl_fu_bead(dirct, subdir,  f'{list_fn[index]}', list_ts[index], k_t=0.03, stat='count', show=False)

    break

bf.cdf_xbead(dirct, figname='', titlestring='', show=False)
bf.violin_trace_vel(dirct, figname='', titlestring='', show=False)
bf.violin_fu_rl(dirct, k_t=0.03, figname='', titlestring='', show=False)
