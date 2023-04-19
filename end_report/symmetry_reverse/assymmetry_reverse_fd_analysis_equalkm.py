from motorgillespie.plotting import cargo_figures as bf

dirct1 = '20221215_132635_symmetry_reverse_fd_equalkm'
ts = [[1,1], [2,2], [3,3], [4,4]]
fdratiolist = [0.75, 0.5, 1.25, 1, 1.5] # HAS TO BE THE RIGHT ORDER!!!!
# 1.5, 0.5, 2.5, 1, 1.5



'''Dataframe'''
#df.rl_bead_n_parratio(dirct=dirct1, filename='', ts_list=ts, km_list=fdratiolist, parname='fd')

'''Bead figures'''
bf.plot_n_parratio_rl(dirct=dirct1, filename='N_parratio_rl.csv', parname='fd', figname='fd', titlestring='', show=True)
