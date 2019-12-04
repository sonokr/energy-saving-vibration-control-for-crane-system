import configparser
import sys

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from scipy import integrate


def load_conditions(cond):
    try:
        config = configparser.ConfigParser()
        config.read('conditions.ini')

        # 時間
        dt = config.getfloat(cond, 'dt')
        Tend = config.getfloat(cond, 'Tend')
        TE = config.getint(cond, 'TE')
        Nrk = round(Tend / dt)
        Nte = round(TE / dt)
        t = np.linspace(0.0, TE, 2*Nte+1)

        # ケーブル
        L = config.getfloat(cond, 'L')
        Lend = config.getfloat(cond, 'Lend')
        dl = Lend - L

        # トロリ
        XS = config.getint(cond, 'XS')
        XE = config.getfloat(cond, 'XE')

        # 重さ
        M = config.getint(cond, 'M')
        m = config.getfloat(cond, 'm_')
    
    except FileExistsError:
        print("Can't Open File.")
        sys.exit()

    values = {
            "dt" : dt,
            "Tend" : Tend,
            "TE" : TE,
            "Nrk" : Nrk,
            "Nte" : Nte,
            "t" : t,
            "L" : L,
            "Lend" : Lend,
            "dl" : dl,
            "XS" : XS,
            "XE" : XE,
            "M" : M,
            "m" : m
    }

    return values

VALUES = load_conditions('cond2')
dt = VALUES['dt']
Tend = VALUES['Tend']
TE = VALUES['TE']
Nrk = VALUES['Nrk']
Nte = VALUES['Nte']
t = VALUES['t']
L = VALUES['L']
Lend = VALUES['Lend']
dl = VALUES['dl']
XS = VALUES['XS']
XE = VALUES['XE']
M = VALUES['M']
m = VALUES['m']


def ufunc1(a):
    a_ = [*a, 1-sum(a)]
    u = np.array([0.0 for i in range(2*Nte+1)])
    for i in range(len(a_)):
        u += a_[i] * (t/TE)**(i+1)
    return u


def ufunc2(a):
    T = -1 + 2*t/TE
    u = t/TE + (1-T**2)*sum([a[n]*T**n for n in range(len(a))])
    return u


def find_inflection_point(x):
    inflection_point = [0, ]
    for i in range(1, len(x)-1):
        if (x[i] >= x[i-1] and x[i] >= x[i+1]) \
           or (x[i] <= x[i-1] and x[i] <= x[i+1]):
           inflection_point.append(i+1)
    if inflection_point[-1] != len(x)-1:
        inflection_point.append(len(x)-1)
    return inflection_point


def energy(f, x):
    p = find_inflection_point(x)
    ene = 0
    for i in range(1, len(p)):
        ene = ene + \
            abs(integrate.simps(
                np.abs(f[p[i-1]:p[i]]), x[p[i-1]:p[i]]))
    return ene


def plot_graph(vals):
    for val, label in vals:
        plt.plot(np.linspace(0.0, Tend, Nrk+1), val)
        plt.xlabel('t [s]')
        plt.ylabel(label)
        plt.show()
