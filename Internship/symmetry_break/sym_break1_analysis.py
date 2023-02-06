from motorgillespie.plotting import cargo_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.analysis import unbiased_walk as uw
from motorgillespie.analysis import print_info as pr
import os

'''Analysis of motor objects obtained from script sym_break1_init.py'''

## Simulation settings ##
dirct = '20221020_112816_sym_break1_symmetry_exp'
team_comb = [(0,1), (0,2), (1,0), (1,1), (2,0), (2,2)] # should be in the right order


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
        bf.rl_fu_bead(dirct, subdir,  f'{list_fn[index]}', list_ts[index], k_t=0.08, stat='count', show=False)

    break

bf.cdf_xbead(dirct, figname='', titlestring='', show=False)
bf.violin_trace_vel(dirct, figname='', titlestring='', show=False)
bf.violin_fu_rl(dirct, k_t=0.08, figname='', titlestring='', show=False)
