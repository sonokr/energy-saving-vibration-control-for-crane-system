import copy
import csv
import pprint
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy import integrate

from common_functions import (energy, find_inflection_point, load_conditions,
                              plot_graph, ufunc1, ufunc2)

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


class Crane:

    def __init__(self):
        pass

    def f(self, x1, x2, l, dl, ddx):
        dx1 = x2
        dx2 = dx2 = -(2*dl*x2 + 9.81*np.sin(x1) + ddx*np.cos(x1)) / l
        return np.array([dx1, dx2])

    def RK4(self, l_, dl_, ddx_):
        k = np.empty([2, 4])
        X = np.zeros([2, Nrk+1])
        for i in range(0, Nrk):
            x1, x2 = X[:, i]
            l, dl, ddx = l_[2*i], dl_[2*i], ddx_[2*i]
            k[:, 0] = self.f(x1, x2, l, dl, ddx)*dt
            k[:, 1] = self.f(x1+k[0, 0]/2.0, x2+k[1, 0]/2.0, l, dl, ddx)*dt
            k[:, 2] = self.f(x1+k[0, 1]/2.0, x2+k[1, 1]/2.0, l, dl, ddx)*dt
            k[:, 3] = self.f(x1+k[0, 2], x2+k[1, 2], l, dl, ddx)*dt
            X[:, i+1] = X[:, i] + ((k[:, 0] + 2.0*k[:, 1] + 2.0*k[:, 2] + k[:, 3]) / 6.0)
        return X

    def trolly(self, a):
        u = ufunc2(a)
        traj = np.zeros([2*Nrk+1, 3])
        traj[:2*Nte+1, 0] = (XE-XS)*(u - (np.sin(2*np.pi*u))/(2*np.pi)) + XS
        traj[2*Nte+1:, 0] = XE
        traj[:, 1] = (np.gradient(traj[:, 0], TE/(2*Nte+1)))
        traj[:, 2] = (np.gradient(traj[:, 1], TE/(2*Nte+1)))
        return traj

    def cable(self):
        traj = np.zeros([2*Nrk+1, 3])
        traj[:2*Nte+1, 0] = L + dl*(t/TE-np.sin(2.*np.pi*t/TE)/(2*np.pi))
        traj[2*Nte+1:, 0] = Lend
        traj[:, 1] = (np.gradient(traj[:, 0], TE/(2*Nte+1)))
        traj[:, 2] = (np.gradient(traj[:, 1], TE/(2*Nte+1)))
        return traj

    def force(self, x1, x2, t, l_):
        x, dx, ddx = t[:, 0], t[:, 1], t[:, 2]
        l, dl, ddl = l_[:, 0], l_[:, 1], l_[:, 2]
        x3 = -(2*dl*x2 + 9.81*np.sin(x1) + ddx*np.cos(x1)) / l
        T = (M+m)*ddx + m*ddl*np.sin(x1) + m*l*x3*np.cos(x1) + 2*m*dl*x2*np.cos(x1) - m*l*x2**2*np.sin(x1)
        return T


def main(args):
    DEFAULT = False

    if DEFAULT:
        a = [-0.0614506, -0.36800381,  0.41197453,  0.21663605, -0.33973749]
    else:
        param_path = './results/{}'.format(args[1])
        with open(param_path) as f:
            reader = csv.reader(f)
            a = [float(row[0]) for row in reader]

    start = time.time() 

    crane = Crane()
    t_traj = crane.trolly(a)
    c_traj = crane.cable()
    vib = crane.RK4(c_traj[:,0], 
                    c_traj[:,1], 
                    t_traj[:,2])
    f = crane.force(vib[0, :], vib[1, :], 
                    t_traj[0:2*Nrk+1:2, :], c_traj[0:2*Nrk+1:2, :])
    x = copy.deepcopy(t_traj[:2*Nte+1:2, 0])
    p = find_inflection_point(x)
    ene = energy(f, x)

    print('係数:')
    pprint.pprint(a)
    print('\n残留振動[deg]: {}'.format(np.amax(abs(180*vib[0, Nte+1:]/np.pi))))
    print('エネルギー[J]: {}'.format(ene))

    plot_graph([[t_traj[0:2*Nrk+1:2, 0], 'x'],
                [180*vib[0, :]/np.pi, 'θ']])

    # 残留振動の加速度とトロリの加速度が打ち消し合うかどうか比較
    x_ = t_traj[0:2*Nrk+1:2,0]
    dx_ = t_traj[0:2*Nrk+1:2,1]
    ddx_ = t_traj[0:2*Nrk+1:2,2]
    l_ = c_traj[0:2*Nrk+1:2,0]
    dl_ = c_traj[0:2*Nrk+1:2,1]
    ddl_ = c_traj[0:2*Nrk+1:2,2]
    s_ = 180*vib[0,:]/np.pi
    ds_ = 180*vib[1,:]/np.pi
    dds_ = (-(2*dl_*ds_ + 9.81*np.sin(s_) + ddx_*np.cos(s_)) / l_)

    plt.title("Condition2 PSO3")
    plt.plot(np.linspace(0.0, TE, 2*Nte+1), (M+m)*ddx_, label="ddx")
    # plt.plot(np.linspace(0.0, TE, 2*Nte+1), m*ddl_*np.sin(s_), label="2")
    plt.plot(np.linspace(0.0, TE, 2*Nte+1), m*l_*dds_*np.cos(s_), label="dds")
    # plt.plot(np.linspace(0.0, TE, 2*Nte+1), 2*m*dl_*ds_*np.cos(s_))
    # plt.plot(np.linspace(0.0, TE, 2*Nte+1), -1*m*l_*ds_**2*np.sin(s_))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main(sys.argv)
