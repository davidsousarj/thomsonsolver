#!/usr/bin/python3
#
# Thomson Problem Solver by @davidsousarj
#
# github.com/davidsousarj/thomsonsolver
#
import sys
import numpy as np
from numpy import sign
from numpy.random import random as rand
from math import sin, cos, sqrt, pi, atan2

def Cart2Sph(x, y, z):
    return sqrt(x*x + y*y + z*z),     \
           atan2(sqrt(x*x + y*y), z), \
           atan2(y, x)

def Sph2Cart(r, theta, phi):
    return r*sin(theta)*cos(phi), \
           r*sin(theta)*sin(phi), \
           r*cos(theta)

def Sph(C):
    return np.array([ Cart2Sph(*n) for n in C ])

def Cart(S):
    return np.array([ Sph2Cart(*n) for n in S ])

def r_ij(r1, r2):
    return sqrt( (r1[0]-r2[0])**2 \
                +(r1[1]-r2[1])**2 \
                +(r1[2]-r2[2])**2 )

def Energy(C):
    r'''Potential Energy of a system of like charges in atomic units.

    E =  \sum_{i=0}^{n} \sum_{j=i+1}^{n} \frac{1}{r_{ij}}
    '''
    E = 0.
    for i in range(len(C)):
        for j in range(i+1,len(C)):
            E += 1. / r_ij(C[i,:], C[j,:])
    return E

def SphEnergy(S):
    E = 0.
    for i in range(len(S)):
        for j in range(i+1,len(S)):
            r,t1,p1 = S[i,0],S[i,1],S[i,2]
            r,t2,p2 = S[j,0],S[j,1],S[j,2]
            E += 1./( sqrt(2)*r*sqrt(1. - cos(t1)*cos(t2) 
                                     - sin(t1)*sin(t2)*cos(p1-p2)))
    return E

def PerturbSphEnergy(S, n, coord, change):
    '''
    Calculates the energy after a small displacement in
    one of the spherical coordinates of the system.
    '''
    S_mod = S[:]
    S_mod[n, coord] += change
    return SphEnergy(S_mod)
    
def GenPoints(n, R):
    '''
    Generate random points on a sphere.
    The First point is always (0,0,R) in cartesians.
    
    To minimize the number of points too close from each other,
    for each new point, M candidates are generated, and the
    farthest from all previous points is selected. 
    '''
    M = n - 2 if n > 2 else 1
    C = np.zeros((n,3))
    C[0] = [0,0,R]
    for i in range(1,n):
        Candidates = np.zeros((M,3))
        Distances  = np.zeros((M))
        for j in range(M):
            Candidates[j] = list(Sph2Cart(R, pi*rand(), 2*pi*rand()))
            for k in range(i):
                Distances[j] += r_ij(C[k, :], Candidates[j])
        
        C[i] = Candidates[ Distances.argmax() ]
    return C

def Iteration(C, E, Step, Print, Out):
    StepLimit = 0.2
    S = Sph(C)
    S_new = S[:]
    for n in range(1,len(S)):
        # E_mod = [E(x,y), E(x-dx,y), E(x+dx,y), E(x,y-dy), E(x,y+dy)]
        E_mod = np.array([E,
                          PerturbSphEnergy(S, n, 1, -Step),
                          PerturbSphEnergy(S, n, 1,  Step),
                          PerturbSphEnergy(S, n, 2, -Step),
                          PerturbSphEnergy(S, n, 2,  Step)])
        if Print > 1: _print(f"{n} {E_mod} {E_mod.argmin()}", Out)
        MinIndex = E_mod.argmin()
        #
        # Numerical evaluation of first and second derivatives
        #
        # Subtracting E_mod[MinIndex] and E_mod[Index2]
        # where the (MinIndex, Index2) pairs are:
        #
        # [[1,2], [2,1], [3,4], [4,3]]
        #
        if MinIndex != 0:
            if MinIndex in (1,3):
                Index2 = MinIndex + 1
            else:
                Index2 = MinIndex - 1
            if MinIndex in (1,2):
                Coord = 1
            else:
                Coord = 2
            
            D1 = (E_mod[MinIndex] - E_mod[Index2]) / (2*Step)
            D2 = (E_mod[MinIndex] -2*E_mod[0] + E_mod[Index2]) / Step**2

            if abs(D1/D2) <= StepLimit:
                S_new[n,Coord] -= D1/D2
            else:        
                S_new[n,Coord] -= sign(D1/D2)*StepLimit        

            if Print > 1: _print(f"{D1} {D2} {D1/D2}\n", Out)

    C_new = Cart(S_new)
    return C_new

def Optimization(C, nRun, n, R, NSim, Tol, Step, MaxIt,
                 OutXYZ, Out, Print, PrintXYZ):            
    E = Energy(C)
    DeltaE = E
    if Print > 0: _print("\n Numerical gradient iterations", Out)
    if Print > 0: _print(" ===================================", Out)
    if Print > 0: _print("  i    Energy          d(Energy)    ", Out)
    i=0
    while abs(DeltaE) > Tol and i <= MaxIt:
        if Print > 0: _print(f" {i:3d}{E:16.8f}{DeltaE:16.8f}", Out)
        C_new = Iteration(C, E, Step, Print, Out)
        E_new = Energy(C_new)
        DeltaE = E_new - E
        i+=1
        C = C_new
        E = E_new

        if PrintXYZ == 2:
            with open(f"{OutXYZ}_{nRun}.xyz", 'a') as f:
                f.write(XYZ(C, PrintXYZ, nRun, i))

    if Print > 0:
        _print(f" {i:3d}{E:16.8f}{DeltaE:16.8f}\n", Out)
    if abs(DeltaE) <= Tol:
        if Print > 0:
            _print(f" Run {nRun:4d} converged in {i:4d} iterations.", Out)
        else:
            _print(f" Run {nRun:4d} converged in {i:4d} iterations. Energy:\
 {E_new:14.8f} a.u.", SYS['Out'])
    else:
        _print(f" Run {nRun:4d} not converged.", Out)
    return C_new, E_new

def _print(text, out):
    if out is None:
        print(text)
    else:
        with open(out + ".out", 'a') as f:
            f.write(text + "\n")

def textC(C, coord='cart'):
    buffer = ""
    l = "="
    if coord == 'cart':
        buffer = "  N      X         Y         Z     \n "
        buffer += l*34 + "\n"
    else: # coord == 'sph':
        buffer = "  N      R        Theta        Phi       \n "
        buffer += l*40 + "\n"
    for n in range(len(C)):
        x,y,z = C[n,0], C[n,1], C[n,2]
        if coord == 'cart':
            buffer += f" {n+1:4d}{x:10.4f}{y:10.4f}{z:10.4f}\n"
        elif coord == 'sph':
            y,z = y/pi, z/pi
            buffer += f" {n+1:4d}{x:10.4f}{y:10.4f} pi{z:10.4f} pi\n"
    return buffer

def DistanceTable(C):
    l = "="
    buffer  = " Distances between points\n " + l*24 + "\n"
    for i in range(len(C)):
        for j in range(i+1,len(C)):
            buffer += f" {i+1}-{j+1} : {r_ij(C[i], C[j]):10.4f}\n"
    return buffer

def XYZ(C, PrintLevel, nRun=0, nIter=0, atom='H'):
    n = len(C)
    E = Energy(C)
    buffer = f"{n}\n"
    if PrintLevel > 0:
        buffer += f" Run: {nRun}"
    if PrintLevel > 1:
        buffer += f" Iter: {nIter}"
    buffer += f" Energy = {E:.8f}\n" 
    for i in range(n):
        x,y,z = C[i,0], C[i,1], C[i,2]
        buffer += f"{atom} {x:12.8f}{y:12.8f}{z:12.8f}\n" 
    return buffer

def PrintHeader(C, t1, t2, Out):
    E = Energy(C)
    _print(f"\n {t1} {t2} geometry (cartesian)", Out)
    Ctext = textC(C)
    _print(Ctext, Out)
    S=Sph(C)
    _print(f" {t1} {t2} geometry (spherical)", Out)
    Stext = textC(S, 'sph')
    _print(Stext, Out)
    _print(DistanceTable(C), Out)
    _print(f" {t1} {t2} Energy: {E:.8f} a.u.", Out)

def convert(data, t):
    '''Convert strings to other data types.'''
    if t == 'int':
        return int(data)
    elif t == 'float':
        return float(data)
    elif t == 'bool':
        return bool(data)
    else: # t == 'str'
        return data

def Configure(args):
    # Parameter Defaults    
    SYS = {'n':None, 'R':1.0, 'NSim':1, 'Tol':1e-6, 'Step':0.02,
           'MaxIt':1000, 'OutXYZ':'Thomson', 'Out':None, 'Print':0,
           'PrintXYZ':0}
    TYPE = {'n':'int', 'R':'float', 'NSim':'int', 'Tol':'float',
            'Step':'float', 'MaxIt':'int', 'OutXYZ':'str',
            'Out':'str', 'Print':'int', 'PrintXYZ':'int'}

    # Help info
    if len(args) > 1 and args[1] == 'help':
        print("Usage: '$ python3 ThomsonSolver.py n=[number of points]\
 [optional arguments=etc.]'")
        print()
        print("Optional arguments:")
        print("R=radius of the sphere in atomic units (default=1.0)")
        print("NSim=Number of simulations to run (default=1)")
        print("Tol=Energy threshold in atomic units (default=1e-6)")
        print("Step=Angle Increment in radians for the numerical\
 gradient calculation (default=0.02)")
        print("MaxIt=Maximum number of iterations (default=1000)")
        print("OutXYZ=Name of output .xyz file (default=Thomson)")
        print("Out=Name of output text file (by default it is\
 printed on screen)")
        print("Print=Print level (default=0)")
        print(" Print=0 means minimal output, just final result.")
        print(" Print=1 prints extra info about each run/iteration.")
        print(" Print=2 prints debug data for each iteration.")
        print("PrintXYZ=Print level for .xyz file (default=0)")
        print(" PrintXYZ=0 creates a .xyz file only for the global minimum.")
        print(" PrintXYZ=1 creates a .xyz for every run.")
        print(" PrintXYZ=2 creates a .xyz for every run, with frames")
        print("            containing every optimization iteration.")
        print()

    # Parsing arguments
    for i in range(1, len(args)):
        token = args[i].split("=")
        if len(token) != 2:
            print("Syntax Error. Please use program as follows:")
            print("$ python3 ThomsonSolver.py n=[number of points]\
 [optional arguments=etc.]")
            print("Use '$ python3 ThomsonSolver.py help' for more\
 details.")
            exit(1)
        if token[0] not in SYS.keys():
            print(f"Error. Parameter {token[0]} not recognized.")
            print("Use '$ python3 ThomsonSolver.py help' for help.")
            exit(1)
        else:
            SYS[token[0]] = convert(token[1], TYPE[token[0]])

    if SYS['n'] is None:
        print(f"Error. Number of points not defined.")
        print("Use '$ python3 ThomsonSolver.py help' for help.")
        exit(1)
    elif SYS['n'] < 2:
        print(f"Error. The minimum number of points is 2.")
        exit(1)
        
    if SYS['Print'] > 1:
        _print(f"System parameters: \n{SYS}", SYS['Out'])

    return SYS

if __name__ == '__main__':
    SYS = Configure(sys.argv)
    if SYS['Out'] is not None:
        open(SYS['Out'] + ".out", 'w').close()

    Coords   = np.zeros((SYS['NSim'], SYS['n'], 3))
    Energies = np.zeros(SYS['NSim'])

    for s in range(SYS['NSim']):    
        C = GenPoints(SYS['n'], SYS['R'])
        if SYS['Print'] > 0:
            PrintHeader(C, f"Run {s}", "Initial", SYS['Out'])

        if SYS['PrintXYZ'] == 2:
            with open(f"{SYS['OutXYZ']}_{s}.xyz", 'w') as f:
                f.write(XYZ(C, SYS['PrintXYZ'], s, 0))

        C_opt, E_opt = Optimization(C, s, **SYS)
        Coords[s] = C_opt
        Energies[s] = E_opt

        if SYS['Print'] > 0:
            PrintHeader(C_opt, f"Run {s}", "Final", SYS['Out'])

        if SYS['PrintXYZ'] == 1:
            with open(f"{SYS['OutXYZ']}_{s}.xyz", 'w') as f:
                f.write(XYZ(C, SYS['PrintXYZ'], s))


    C_min = Coords[ Energies.argmin() ]
    PrintHeader(C_min, "Global", "Minimum", SYS['Out'])
    if SYS['PrintXYZ'] > -1:
        with open(f"{SYS['OutXYZ']}_global.xyz", 'w') as f:
            f.write(XYZ(C_min, SYS['PrintXYZ']))
