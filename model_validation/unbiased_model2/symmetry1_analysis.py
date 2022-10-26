from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
from motorgillespie.analysis import unbiased_walk as uw
from motorgillespie.analysis import print_info as pr
import os

'''Analysis of motor objects obtained from script symmetry1_init.py'''

## Simulation settings ##
dirct = '20221011_153538_symmetry1.2'
team_comb = [(1,1), (2,2)] #should be in the right order
eps = ['constant', 'exponential', 'gaussian'] #should be in the right order

# Create lists with
list_fn = []
list_ts = []
for i in team_comb:
    for j in eps:
        list_fn.append(f'{i}_{j}')
        list_ts.append(f'kin,dyn={i}; unbind eq.={j}')

for root, subdirs, files in os.walk(f'.\motor_objects\\{dirct}'):
    for index, subdir in enumerate(subdirs):
        pr.inspect(dirct, subdir)
        bf.xbead_dist(dirct, subdir, f'xb_dist_{list_fn[index]}', list_ts[index], interval=(0, 90), show=False)
        mf.xm_dist(dirct, subdir, f'xm_dist_{list_fn[index]}', list_ts[index], show=False)
        mf.forces_dist(dirct, subdir, f'motorforce_dist_{list_fn[index]}', list_ts[index], show=False)
        bf.cdf_xbead(dirct, subdir,  f'CDF_xbead_{list_fn[index]}', list_ts[index], interval=(0, 95), stepsize=0.001)
        uw.intrpl_bead_symmetry(dirct, subdir, list_ts[index])

    break
