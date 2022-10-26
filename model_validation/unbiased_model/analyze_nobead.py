from model_validation.unbiased_model import event_statistics as es
from motorgillespie.analysis import unbiased_walk as uw

'''Analysis of motor objects obtained from script: init_sym for different settings indicated beneath'''

## Settings ##
dirct = 'model_validation'
subdir = '20221003_113514_constalfa'
varname = 'kt'
list_kt = [0.08]


for varvalue in list_kt:
    figname = f'{varvalue}{varname}'
    #mf.forces_dist(dirct, subdir, varname,  varvalue, figname, interval=(0, 90), stepsize=0.001, stat='probability')
    #bf.xbead_dist(subdir, varvalue, interval=(0, 1.5), stepsize=0.001, subject=dirct, stat='probability')

    uw.intrpl_bead_symmetry(subdir, varvalue, subject=dirct)
    #uw.bead_symmetry(subdir, varvalue, subject=dirct)

    #es.xbead_stats(dirct, subdir, varname, varvalue, interval=(0, 8), stepsize=0.001, hypothesis='norm')
    es.fair_event_choice(dirct, subdir, varname, varvalue, p=0.5, alt='two-sided')
