# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

'''

2.
###- Gaussian function for unbinding rate OR quadratic function, with shift CHECK
    ###- Change trap stiffness (if K_t is large, you get large forces, and then large stall times?) CHECK
    ###- Plot P(F_u)CHECK and P(F)CHECK>>Then first normalize over time!!CHECK distributions + trajectories CHECK + unbinding events per F_u per k_t CHECK and Tau distribution(from 250ms!!)CHECK
    ###- Correct for 0 force for unbound motor CHECK

26-08-2022
3.
- Interpolate lists with 'previous' option to make trajectories like data CHECK
> np.rand.gaussian(mean=0, var=look at experimental data, shape =len(t_new)) >> add this to x_bead_interplated list CHECK
- Also use interpolate instead of np.diff, then you have new lists of times and locations that are normalized and countable by the distribution plot CHECK
- Make figures without interpolation to check if interpolated distributions make sense CHECK

- ** Again check outputted data, automated! (Think about print() lines, breakpoints(), AssertionErrors(like the force_netto, match_events, etc), manually check data from motor objects by using a function that outputs/prints important stats/counts/conditions/whole lists >> think about what is important and put in function.
    > So think about 1. Physical properties (like net_force) 2. Script functionality (like match_events and default/keyword arguments, but also think about how to check if the right info is collected/appended to lists)
    > 3. Figure default parameters 4. If functions you use do what you think they do (like np.rand, np.interp1d)
- ** Think about nice figures >> Look at articles (maybe contour plots, because now you are looking at one variable at a time) CHECK
        >> Of course think about what you are actually investigating CHECK
        >> Streamline everything in the same format, remove any hardcoding, and make adjustable as much as possible, otherwise make same function for other conditions, make normal short names!!) CHECK
- ** Think about what is relevant if you think about a team of motors where the bead is mearured (so what data is interesting: example: individual motor force distribution, coloured per motor and per variable(like Kt, or team size)) CHECk


4. UNBIASED
4.1 Symmetric simulation: two kinesins, bead (varvalue=0), epsilon=0 > make bead trajectory histogram (should be unbiased, random motion, brownian) CHECK
>> First normalize the time!! You have lists of time points and Xb's. CHECK
> They should stall at 7pN until simulation is over CHECK
>> Check again the order! Analyze with statistics check if there is a bias CHECK(there is) >> f_s = 10000
4.2 Add Kt, then add epsilon(constant) > make bead traj. histograms again for these settings CHECK
>> Think about how binding and unbinding would affect those histograms CHECK
    >  With Kt and no epsilon, also stalls at 7pN until simulation is over, but less diffusion, biased to the center (x=0) CHECK
4.3  Also try epsilon + no Kt >  use this extreme case to check your code: after unbinding Xb should equal Xm, and so the force 0 CHECK
>> and thus epsilon0 + alfa0 CHECK (not the case, but found reason)
4.4 EVERYTHING UNBIASED?? check
oplossingen:
1. Biological relevant force range (so just 7pN), this is solved by just including unbinding and rebinding CHECK
2. CHECK Try to objects moving apart, without force calculated by bead location, then you know if it's the bead, and not the event drawing and/or motor order bias (maybe still calculate if it's an unbiased coin, with both retro and antero motor first in list)
3. Check if distribution is normal with KS test (!!!), i.e are simulated distributions same as a true gaussian? CHECK



5. FUNTIMES
5.1 D(K_t) >> how is the diffusion dependent on Kt? (high k_t makes tales of gaussian shorter) >> variance as a function of k_t
5.2 Think about how to include different species of motor proteins to simulate a tug of war CHECK
5.3 Use the KS distribution test CHECK to see if data are from different distributions > change parameters of certain motors in symmetry1 runs
++ Make distributions of experimental data in Matlab CHECK, also analyze those with the KS test in comparason with above
5.4 See if symmetry1 breaks with two different k_m CHECK
5.5 Investigate different unbinding equations CHECK
5.6 Make CDF's
5.7 Discrete differentating to calculate V > should equal V0 at beginning
5.8 Zie onder aan dit script


 Reading/checking
 ( Read article 70nm bead and crispr article Anna Akmanova )
 1. Read about tracking motors live in vivo
 2. Search for rest_length by reading about kinesin structural biology
 3. Read about polymere stretch behaviour (for motor proteins specific if possible)
 4. Check if plateuas are seen under all conditions (like ATP concentration) or kinesin species


'''
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/




'''
>> Model
# Think about including back stepping (P(forward) = P(backwards) at Fs)
# Add angle
> Fx and Fy and how the rates depend on those
> How does this work for >2 motors?
> Bead/Cargo size
(> Think about dynamics of motor proteins on cargo)

>> Programming
# Checkpoint lines for values of: variables and constants >> DEBUGGING lines
(# Getters and setters)
# Check for bottlenecks in runtime 
# All parameters in same order in all the modules
# All parameter names same between functions and modules
# See what is a good way of naming files and directories 
1. Think about the CL tools from courses you had
2. However, also think wat is the fastest for you (because that is different from what is handy according to 1.)

>> Math
(# Write out equations)
# Do the fitting of parameters on analytical eqs (Fu and X walk) with data from exponential >Matlab?
# Solve equations in Mathematica for exponential epsilon > then plot that equation against simulated data with exponential epsilon 
(# Take derivative of displacement over discrete time)'''
