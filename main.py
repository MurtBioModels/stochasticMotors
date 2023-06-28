# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/




'''
>> Model
# Think about including back stepping (P(forward) = P(backwards) at Fs) N.A.
# Add angle DONE
> Fx and Fy and how the rates depend on those DONE
> How does this work for >2 motors? N.A.
> Bead/Cargo size DONE
(> Think about dynamics of motor proteins on cargo) THOUGHT ABOUT IT

>> Programming
$ # Checkpoint lines for values of: variables and constants >> DEBUGGING lines
(# Getters and setters) CHECK/N.A.
# Check for bottlenecks in runtime CHECK
$ # All parameters in same order in all the modules 
$ # All parameter names same between functions and modules
# See what is a good way of naming files and directories (BIG)CHECK
1. Think about the CL tools from courses you had YES
2. However, also think wat is the fastest for you (because that is different from what is handy according to 1.) YES

- ** Again check outputted data, automated! (Think about print() lines, breakpoints(), AssertionErrors(like the force_netto, match_events, etc), manually check data from motor objects by using a function that outputs/prints important stats/counts/conditions/whole lists >> think about what is important and put in function.
    > So think about 1. Physical properties (like net_force) 2. Script functionality (like match_events and default/keyword arguments, but also think about how to check if the right info is collected/appended to lists)
    > 3. Figure default parameters 4. If functions you use do what you think they do (like np.rand, np.interp1d)
- ** Think about nice figures >> Look at articles (maybe contour plots, because now you are looking at one variable at a time) CHECK
        >> Of course think about what you are actually investigating CHECK
        >> Streamline everything in the same format, remove any hardcoding, and make adjustable as much as possible, otherwise make same function for other conditions, make normal short names!!) CHECK
- ** Think about what is relevant if you think about a motor_team of motors where the bead is mearured (so what data is interesting: example: individual motor force distribution, coloured per motor and per variable(like Kt, or motor_team size)) CHECk

>> Math
(# Write out equations)
# Do the fitting of parameters on analytical eqs (Fu and X walk) with data from exponential epsilon >Matlab?
# Solve equations in Mathematica for exponential epsilon > then plot that equation against simulated data with exponential epsilon 
(# Take derivative of displacement over discrete time)'''
