import sys
import numpy
import sympy
import mpmath
import itertools
import pandas as pd
import multiprocessing as mp

from sympy import re, I
from sympy import *
from sympy.abc import *
from datetime import datetime
from scipy.optimize import minimize, fsolve
from numpy.lib.scimath import sqrt as csqrt

def FreeEnergyTI(param):
    def _h0(t):
        if t<1:
            h0 = (t*(numpy.arccos(t)-t*csqrt(1-t**2))) / (1-t**2)**(3/2)
        if t>1:
            h0 = (re(t*(-numpy.arccosh(t)+t*csqrt(t**2-1))) / (-1+t**2)**(3/2))
        if td == 1:
            h0 = 2/3
        return(h0)

    def _h(td):
        if td<1:
            h  = (td*(numpy.arccos(td)-td*csqrt(1-td**2))) / (1-td**2)**(3/2)
        if td>1:
            h  = (re(td*(-numpy.arccosh(td)+td*csqrt(td**2 -1))) / (-1+td**2)**(3/2))
        if td == 1:
            h  = 2/3
        return(h)

    def Mag(EH):
        #Magnitization depends on effective H (EH)
        Mag = Ms * sympy.tanh(X * EH / Ms)
        return Mag

    def EH3():
        #finding root of given EH equation at H
        #we format into mpmath to use its findroot function
        _ex = .001
        _ey = .001
        _ez = .001
        eReplace = [(ex, _ex), (ey, _ey), (ez, _ez)]
        EH_eq = H-D*Mag(EH) + f*replacedLz.subs(eReplace)*Mag(EH) / ((1+_ex)*(1+_ey)*(1+_ez)) - EH
        #This method lets us start at H in solving, may more accurate, but it is slower I think.
        lam_EH3 = lambdify(EH, EH_eq, 'scipy')
        EH3 = fsolve(lam_EH3, H)
        #EH3 = sympy.solvers.solve(EH_eq)
        #EH3 = mpmath.findroot(lam_EH3, H, solver = 'mdnewton', norm = 'max-norm')
        if len(EH3) > 1:
            print('Warning, solving for EH3 gives more than 1 solution: ' + str(EH3))
            raise
        EH3 = float(EH3[0])
        print('\t H:   ' + str(H))
        print('\t EH3: ' + str(EH3))
        return EH3

    print('defining params')
    #import pdb; pdb.set_trace()
    #param = list(param)
    H  = param[0]
    Ms = param[1]
    X  = param[2]
    t  = param[3]
    td = param[4]
    f  = param[5]
    nu = param[6]
    g  = param[7]
    h  = _h(td)
    h0 = _h0(t)
    
    print('defining variables')
    Sigma = 2*g*(1+nu)
    k     = Sigma / (2*(1+nu)*(1-2*nu))
    l     = (Sigma*nu) / (2*(1+nu)*(1-2*nu))
    n     = Sigma*(1-nu) / (2*(1+nu)*(1-2*nu))
    m     = Sigma / (2*(1+nu))
    p     = Sigma / (2*(1+nu))
    k1    = 2*g*(1+nu) / (1-2*nu)
    kp    = ((7*h0-2*t**2-4*t**2*h0)*g+3*(h0-2*t**2+2*t**2*h0)*k1) / (8*(1-t**2)*g*(4*g+3*k1))
    lp    = ((g+3*k1)*(2*t**2-h0-2*t**2*h0)) / (4*(1-t**2)*g*(4*g+3*k1))
    np    = ((6-5*h0-8*t**2+8*t**2*h0)*g+3*(h0-2*t**2+2*t**2*h0)*k1) / (2*(1-t**2)*g*(4*g+3*k1))
    mp    = ((15*h0-2*t**2-12*t**2*h0)*g+3*(3*h0-2*t**2)*k1) / (16*(1-t**2)*g*(4*g+3*k1))
    pp    = (2*(4-3*h0-2*t**2)*g+3*(2-3*h0+2*t**2-3*t**2*h0)*k1) / (8*(1-t**2)*g*(4*g+3*k1))
    kd    = ((7*h-2*td**2-4*td**2*h)*g+3*(h-2*td**2+2*td**2*h)*k1) / (8*(1-td**2)*g*(4*g+3*k1))
    ld    = ((g+3*k1)*(2*td**2-h-2*td**2*h)) / (4*(1-td**2)*g*(4*g+3*k1))
    nd    = ((6-5*h-8*td**2+8*td**2*h)*g+3*(h-2*td**2+2*td**2*h)*k1) / (2*(1-td**2)*g*(4*g+3*k1))
    md    = ((15*h-2*td**2-12*td**2*h)*g+3*(3*h-2*td**2)*k1) / (16*(1-td**2)*g*(4*g+3*k1))
    pd    = (2*(4-3*h-2*td**2)*g+3*(2-3*h+2*td**2-3*td**2*h)*k1) / (8*(1-td**2)*g*(4*g+3*k1))

    ax,ay,az,ex,ey,ez,EH=symbols('ax ay az ex ey ez EH',real=True)
    
    #ax = td*((1+ex)/(1+ez))
    #ay = td*((1+ey)/(1+ez))
    #az = td*((1+ez)/(1+ex))
    #     Lx = (1-1/(1-ax**2)*(1-ax/sqrt(1-ax**2)*asin(sqrt(1-ax**2)))/2)
    #     Ly = (1-1/(1-ay**2)*(1-ay/sqrt(1-ay**2)*asin(sqrt(1-ay**2)))/2)
    #     Lz = (1 /(1-az**2)*(1-az/sqrt(1-az**2)*asin(sqrt(1-az**2))))
    #     #import pdb; pdb.set_trace()
    #     Lx = Lx.subs([(ax,td*((1+ex)/(1+ez)))])
    #     Ly = Ly.subs([(ay,td*((1+ey)/(1+ez)))])
    #     Lz = Lz.subs([(az,td*((1+ez)/(1+ex)))])

    Lx = Function('Lx')(ex, ey, ez)
    Ly = Function('Ly')(ex, ey, ez)
    Lz = Function('Lz')(ex, ey, ez)
    Lreplace = [
    (Lx, 1-1/(1-ax**2)*(1-ax/re(sqrt(1-ax**2))*asin(re(sqrt(1-ax**2))))/2),
    (Ly, 1-1/(1-ay**2)*(1-ay/re(sqrt(1-ay**2))*asin(re(sqrt(1-ay**2))))/2),
    (Lz, 1/(1-az**2)*(1-az/re(sqrt(1-az**2))*asin(re(sqrt(1-az**2)))))
    ]
    aspectReplace = [(ax, td*((1+ex)/(1+ez))),
    (ay, td*((1+ey)/(1+ez))),
    (az, td*((1+ez)/(1+ex)))
    ]
    #We preserve L's for derivatives later, alser calculating L's this way makes minimization faster (idk why)
    replacedLx = Lz.subs(Lreplace).subs(aspectReplace)
    replacedLy = Lz.subs(Lreplace).subs(aspectReplace)
    replacedLz = Lz.subs(Lreplace).subs(aspectReplace)

    print('defining matrices')
    L1 = numpy.matrix([[k1+(4*g)/3, k1-(2*g)/3, k1-(2*g)/3, 0, 0, 0], [k1-2*g/3, k1+4*g/3, k1-2*g/3, 0, 0, 0], [k1-2*g/3, k1-2*g/3, k1+4*g/3, 0, 0, 0], [0, 0, 0, g, 0, 0], [0, 0, 0, 0, g, 0], [0, 0, 0, 0, 0, g]])
    Pi = numpy.matrix([[kp+mp, kp-mp, lp, 0, 0, 0], [kp-mp, kp+mp, lp, 0, 0, 0], [lp, lp, np, 0, 0, 0], [0, 0, 0, 2*pp, 0, 0], [0, 0, 0, 0, 2*pp, 0], [0, 0, 0, 0, 0, 2*mp]])
    Pd = numpy.matrix([[kd+md, kd-md, ld, 0, 0, 0], [kd-md, kd+md, ld, 0, 0, 0], [ld, ld, nd, 0, 0, 0], [0, 0, 0, 2*pd, 0, 0], [0, 0, 0, 0, 2*pd, 0], [0, 0, 0, 0, 0, 2*md]])
    P = numpy.matrix(1/(1-f)*(Pi-f*Pd), dtype = 'float')
    Ltilde = L1+f/(1-f)*numpy.linalg.inv(P)
    
    #import pdb; pdb.set_trace()
    D = numpy.real((4*numpy.pi)/(t**2-1)*(t/(2*csqrt(t**2-1))*numpy.log((t+csqrt(t**2-1))/(t-csqrt(t**2-1)))-1))
    C = numpy.matrix([[Ltilde[0,0], Ltilde[0,1], Ltilde[0,2]], [Ltilde[1,0], Ltilde[1,1], Ltilde[1,2]], [Ltilde[2,0], Ltilde[2,1], Ltilde[2,2]]])
    MTI = (sympy.Matrix([[ex, ey, ez]]) * C * sympy.Matrix([[ex], [ey], [ez]]))[0]
    
    #Equation for free energy equation in transverse isotropy. Reference NIST notebook for equation.
    MagEH3 = Mag(EH3())
    #Ls = ((f*D*MagEH3**2)/2)*((replacedLx + replacedLy + replacedLz)/((1+ex)*(1+ey)*(1+ez)))
    #LsMTI = ((f*D*MagEH3**2)/2)*((replacedLx + replacedLy + replacedLz)/((1+ex)*(1+ey)*(1+ez))) + MTI/2
    FTI = -f*H*MagEH3 + (((-2*f**2)*MagEH3**2)/2)*((replacedLx + replacedLy + replacedLz)/((1+ex)*(1+ey)*(1+ez))) + MTI

    print('minimizing')
    strains = [ex, ey, ez]
    FTI_lam = lambdify(strains, FTI, modules = 'numpy')
    def vectorizeFTI(_list):
        return FTI_lam(_list[0], _list[1], _list[2])
    strains0 = numpy.array([0,0,0])
    minimization = minimize(fun = vectorizeFTI, x0 = strains0, method = 'Nelder-Mead');
    print(str(list(param) + list(minimization.x)))
    return list(param) + list(minimization.x)

def main(params):
    S = []
    for param in params:
        S.append(FreeEnergyTI(param))
    return S
        