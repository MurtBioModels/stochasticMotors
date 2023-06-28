from analysis_plotting import stat_test as st
from analysis_plotting import indlude_in_report as ir

'''Analysis of motor objects obtained from script km_symbreak_init..py'''

dirct1 = '20230411_130059_100_False_notbound'
tslist = [[1, 1], [2, 2], [3, 3], [4, 4]]
km_minus_list = [0.15, 0.1, 0.25, 0.2, 0.35, 0.3, 0.4] # HAS TO BE THE RIGHT ORDER!!!!
ts_include = ['[1, 1]', '[2, 2]', '[3, 3]', '[4, 4]']

st.plot_n_kmr_unbindevent(dirct=dirct1, filename='unbindevents_Nkmr_.csv', n_include=ts_include, show=True)

## CARGO ##
# Run Length
ir.rl_cargo_n_kmr(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, filename='')
ir.plot_n_kmratio_rl(dirct=dirct1, filename='rl_cargo_Nkmr_.csv', n_include=ts_include, km_include=km_minus_list, show=False, figname='')

# Bind time
ir.bt_cargo_n_kmr(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, filename='')
ir.plot_n_kmratio_bt(dirct=dirct1, filename='bt_Nkmr_.csv', n_include=ts_include, km_include=km_minus_list, show=False, figname='')

# xb
ir.xb_n_kmr_2(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, stepsize=0.1, samplesize=100, filename='')
ir.plot_N_kmr_xb_2(dirct=dirct1, filename='xb_Nkmr_.csv', n_include=ts_include, km_include=km_minus_list, stat='probability', show=False, figname='')
ir.plot_N_kmr_xb_2_cdf(dirct=dirct1, filename='xb_Nkmr_.csv', n_include=ts_include, km_include=km_minus_list, show=True, figname='')

# trajectories
ir.traj_n_kmr(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, show=False)

# Bound motors
ir.boundmotors_n_kmr(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, stepsize=0.1, filename='')
ir.plot_n_kmr_boundmotors(dirct=dirct1, filename='anteroretrobound_Nkmratio_.csv', n_include=ts_include, km_include=km_minus_list, show=False, figname='')

# Motor unbinding events
ir.unbindevent_n_kmr(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, filename='')
ir.plot_n_kmr_unbindevent(dirct=dirct1, filename='unbindevents_Nkmr_.csv', n_include=ts_include, km_include=km_minus_list, show=False, figname='')

# Segments
ir.segment_n_kmr(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, filename='')
ir.plot_n_kmr_seg(dirct=dirct1, filename='', n_include=ts_include, km_include=km_minus_list, stat='probability', show=False, figname='')
ir.seg_back_n_kmr(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, filename='')
ir.plot_n_kmr_seg_back(dirct=dirct1, filename='', n_include=ts_include, km_include=km_minus_list, stat='probability', show=False, figname='')

# motor forces
ir.motorforces_n_kmr_2_sep(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, stepsize=0.1, samplesize=40, filename='ss40')
ir.plot_N_kmr_forces_motors_pdf_sep(dirct=dirct1, filename='motorforces_sep_Nkmminus__.csv', n_include=ts_include, km_include=km_minus_list, stat='probability', show=False, figname='')

ir.motorforces_n_kmr_2(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, stepsize=0.1, samplesize=50, filename='')
ir.plot_N_kmr_forces_motors_cdf(dirct=dirct1, filename='N_kmratio_motorforces_.csv', n_include=ts_include, km_include=km_minus_list, show=False, figname='')
ir.plot_N_kmr_forces_motors_violin(dirct=dirct1, filename='N_kmratio_motorforces_.csv', n_include=ts_include, km_include=km_minus_list, show=False, figname='')

# motor displacement
ir.xm_n_kmr_2_sep(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, stepsize=0.1, samplesize=100, filename='')
ir.plot_N_kmr_xm_sep(dirct=dirct1, filename='xm_sep_Nkmminus_.csv', n_include=ts_include, km_include=km_minus_list, stat='probability', show=False, figname='')

# motor runlengths
ir.rl_motors_n_kmr_sep(dirct=dirct1, ts_list=tslist, kmminus_list=km_minus_list, filename='')
ir.plot_n_kmratio_rl_motors_sep(dirct=dirct1, filename='rl_cargo_Nkmr_.csv', n_include=ts_include, km_include=km_minus_list, show=False, figname='')


