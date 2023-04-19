from motorgillespie.plotting import cargo_figures as bf

dirct1 = '20221214_143601_symmetry_reverse_fs_equalkm'
ts = [[1,1], [2,2], [3,3], [4,4]]
fsratiolist = [0.5, 0.57, 0.71, 0.88, 1] # HAS TO BE THE RIGHT ORDER!!!!




'''Dataframe'''
#df.rl_bead_n_parratio(dirct=dirct1, filename='', ts_list=ts, km_list=fsratiolist, parname='fs')

'''Bead figures'''
bf.plot_n_parratio_rl(dirct=dirct1, filename='N_parratio_rl.csv', parname='fs', figname='fs', titlestring='', show=False)
