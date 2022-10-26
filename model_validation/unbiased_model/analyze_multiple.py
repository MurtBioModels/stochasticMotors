from model_validation.unbiased_model import event_statistics as es

'''Analysis of motor objects obtained from script: init_sym for different settings indicated beneath'''

## Settings ##
dirct = 'model_validation'
subdir = '20221003_145915_multiple'
varname = 'kt'
list_kt = [0.08]


for varvalue in list_kt:
    figname = f'{varvalue}{varname}'

    es.fair_first_step(dirct, subdir, interval=(0, 9), stepsize=0.001, n_exp=100, p=0.5, alt='two-sided')
