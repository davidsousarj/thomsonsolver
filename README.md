# thomsonsolver
A simple solver for the Thomson Problem in Python (https://en.wikipedia.org/wiki/Thomson_problem).

### Requisites: 
Python and NumPy library.

### Usage:
`$ python3 ThomsonSolver.py n=[number of points] [optional arguments=etc.]`

###Optional arguments:
- `R=[floating-point number]`: radius of the sphere in atomic units (default=1.0)
- `NSim=[integer]`: Number of simulations to run (default=1)
- `Tol=[number]`: Energy threshold in atomic units (default=1e-6)
- `Step=[floating-point number]`: Angle Increment in radians for the numerical gradient calculation (default=0.02)
- `MaxIt=[integer]`: Maximum number of iterations (default=1000)
- `OutXYZ=[string]`: Name of output `.xyz` file (default=`'Thomson'`)
- `Out=[string]`: Name of output text file (by default it is printed on screen)
- `Print=[integer]`: Print level (default=0)
    - `Print=0` means minimal output, just final result.
    - `Print=1` prints extra info about each run/iteration.
    - `Print=2` prints debug data for each iteration.
- `PrintXYZ=[integer]`: Print level for `.xyz` file (default=0)
    - `PrintXYZ=0` creates a `.xyz` file only for the global minimum.
    - `PrintXYZ=1` creates a `.xyz` for every run.
    - `PrintXYZ=2` creates a `.xyz` for every run, with frames containing every optimization iteration.
 
### Example:
`$ python3 ThomsonSolver.py n=5 Print=0 PrintXYZ=0 NSim=100 Out=report_5 OutXYZ=system_5`

It will solve the Thomson problem for the case n = 5, by running 100 simulations starting from a random configuration of points and doing an approximate Newton-Raphson optimization to achieve the minimal energy configurations. The information about whether each run was converged and the final energies will be shown in the text file `report_5.out` and the coordinates of the global minimum (the optimized configuration with the lowest energy) will be printed to the file `system_5.xyz`, which can be visualized in computational chemistry programs like Avogadro, JMol, or wxMacMolPlt (we recommend the last one because it can show an Energy Plot of the optimization, if you set `PrintXYZ=2`).

Since the coordinates are randomly generated, the results may vary. I have run the above command and got the global minimum energy = 6.47479497 a.u., which is very close to the exact solution, 6.474691495 a.u.  
