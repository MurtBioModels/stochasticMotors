from motorgillespie.plotting import bead_figures as bf
from motorgillespie.plotting import motor_figures as mf
import os
import pickle

# DINGEN PROBEREN HIER
import os.path
path='.\motor_objects\\20221101_123744_elastic_coupling_withtrap'
for root,subdir,files in os.walk(path):
   for name in subdir:
       print('PRINT NAME IN SUBDIR')
       print(os.path.join(path,name))
       name2 = os.path.join(path,name)
       for root,subdir2,files2 in os.walk(name2):
           for name3 in files2:
               print('PRINT NAME IN FILES')
               print(os.path.join(path,name3))

'''Analysis of motor objects obtained from script teamsize_km_symbreak1_init.py_init.py'''

'''
## Simulation settings ##
#dirct = '20221031_133121_teamsize_km_symbreak1'
team_comb = [(3,3)]
#retro_km = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
retro_km = [0.02]

# Create lists with
list_fn = []
list_ts = []

for j in retro_km:
    list_fn.append(f'(1,1)_{j}')
    list_ts.append(f'kin,dyn=(1,1)_{j}k_m')


for root, subdirs, files in os.walk(f'.\motor_objects\{dirct}'):
    for index, subdir in enumerate(subdirs):
        if subdir == 'figures':
            continue
        print(index)
        print(f'subdir={subdir}')
        team = []
        for root, subdirs, files in os.walk(f'.\motor_objects\{dirct}\{subdir}'):
            for filename in files:
                print(f'filename={filename}')
                if filename == 'motor0' or filename == 'parameters.txt':
                    continue
                pickle_motor = open(f'.\motor_objects\\{dirct}\{subdir}\{filename}', 'rb')
                motor = pickle.load(pickle_motor)
                pickle_motor.close()
                team.append(motor)
        print(len(team))
        for motor in team:
            print(f'{motor.id}_{motor.family}_{motor.direction}')
        #bf.xbead_dist(dirct, subdir, f'{list_fn[index]}', list_ts[index], show=False)
        #mf.xm_dist(dirct, subdir, f'{list_fn[index]}', list_ts[index], show=False)
        #mf.forces_dist(dirct, subdir, f'{list_fn[index]}', list_ts[index], show=False)
        #bf.trace_velocity(dirct, subdir, f'{list_fn[index]}', list_ts[index],  stat='count', show=False)
        #bf.rl_fu_bead(dirct, subdir,  f'{list_fn[index]}', list_ts[index], k_t=0.0000001, stat='count', show=False)

    break


#bf.cdf_xbead(dirct='20221025_164036_teamsize_km_symbreak1_(1,1)', figname='', titlestring='', show=False)
#bf.violin_xb(dirct=dirct, figname='', titlestring='', stepsize=0.001, show=False)
#bf.violin_trace_vel(dirct, figname='', titlestring='', show=False)
#bf.violin_fu_rl(dirct, k_t=0.0000001, figname='', titlestring='', show=False)

'''

