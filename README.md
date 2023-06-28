# 2DMotorGillespie

This README file is not like a normal README file, but specially
created for my Major Research Project of the MSc programme 
Bioinformatics and Biocomplexity of Utrecht University, and therefore
specifically directed to dr. Florian Berger. As this project is meant
to be generally used, but also, in the current version, meant
for my own use in my project. If a newer version comes out,
this file will be replaced for general use.

## Project structure
- The 'motorgillespie' file comprised all the modules that make up
the Gillespie simulation tool.
- 'analysis_plotting' contains all the functions I used for plotting and analysing
This should not be considered as part of the tool, as I did not make
it very general and mostly for my own use. The documentation also isn't complete
because that felt a bit too much.

**'indlude_in_report.py' contains all
functions that create the dataframes and figures included (and more) in my report!!**
- 'end_report' contains all the script that produced the data used in my report
The motor objects are too large in size, so I can zip and email them if you want to
take a look at them!

**For creating new script, this is only possible within the root directory
of this project! So create a directory like 'end_project'**.

## motorgillespie tool
- 'gillespie_simulation.py' is the actual Gillespie simulation
- 'gs_functions.py' contains supporting functions used by 'gillespie_simulation.py'
- 'initiate_motors.py' initiates the motors.
- 'motor_class.py' contains the classes used by 'initiate_motors.py'
- 'variable_loops.py' automates everything by using 'initiate_motors.py'
and 'gillespie_simulation.py'

The scripts in 'end_report' can be used as a template to show how
the simulation tool can be used.

### Future improvements
- Flexible paths, so create files outside of the project root directory.
This felted a bit of a waste of time since I was the only one using the project,
and I felt it was better to use my time for things that would help my internship.
- The external force 'f_ex' now had to be parsed as a negative value.
This is fine as it is added to the numerator for calculating the cargo displacement,
but it should be a positive value which is substracted in the numerator.
I remembered I actually did it for a reason, but I don't remember the reason,
to make sure I don't break te code at the last minute I keep it this way.
It should be corrected because it is confusing.
- I changed some attribute names of the classes, but because the pickled
objects still had the old names, this creates errors in the figure fucnctions.
I did add comments there which explain how to change it.
- When saving the objects, the order in which they are stored is dependent
on the decimals in the name and the filesystem of the machine used,
it is commented in the scripts within 'end_report', but should defenitely
be fixed when used by others.

## Testing
I never learned how to test code in a professional and structured way,
and learned this throughout the project. Hence, a lot is tested by
ad hoc solutions like printing, breakpoints, test runs etc.
All those print statements are gone because they should be in a final code,
but if you have any questions about how I tested certain aspects,
just contact me, I remember how I tested everything so I can explain it!
I did try to start redoing the ad hoc testing with unittests, but I only
redid a small part.

## What did I learn during the internship?
I can summarise it as, before the internship, I knew as much about programming
as the students from the Biological Physics course. I learned myself about
OOP (I would say advanced), testing, debugging making libraries, making commandline tools, project structures,
documentation, file handling, efficient memory handling, improving speed,
assert statements, exception handling, more advanced plotting, control version (git),
more advanced functions and probably a lot more.

