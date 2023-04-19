from motorgillespie.plotting import cargo_figures as bf

dirct1 = '20221219_164554_symmetry_reverse_fs3.5vs7'
ts = [[1,1], [2,2], [3,3], [4,4]]
minus_km = [0.04, 0.06, 0.08, 0.12, 0.14, 0.16, 0.18, 0.1, 0.2] # HAS TO BE THE RIGHT ORDER!!!!




'''Dataframe'''
#df.rl_bead_n_parratio(dirct=dirct1, filename='', ts_list=ts, km_list=minus_km, parname='minus_km_notratio')

'''Bead figures'''
bf.plot_n_parratio_rl(dirct=dirct1, filename='N_parratio_rl.csv', parname='km_minus', figname='km_minus', titlestring='', show=True)
