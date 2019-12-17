import configparser
import copy
import random
import sys
import time

import numpy as np
from numba import jit
from scipy import integrate
from tqdm import tqdm

from common_functions import energy, find_inflection_point, load_conditions
from simulator import Crane

VALUES = load_conditions('cond1')
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


def error1(vib):
    return np.amax(abs(180*vib[0, Nte+1:]/np.pi))

def error2(vib):
    return np.sum(abs(vib))

def error3(obj, t_traj, c_traj, vib):
    f = obj.force(vib[0, :],
                  vib[1, :],
                  t_traj[0:2*Nrk+1:2, :],
                  c_traj[0:2*Nrk+1:2, :])
    x = copy.deepcopy(t_traj[:2*Nte+1:2, 0])
    ene = energy(f, x)
  
    K = 1.0
    return ene+K*error1(vib), ene, error1(vib)


class PSO:

    def __init__(self):
        self.obj = Crane()
        self.c_traj = self.obj.cable()

        self.progress = np.empty([200, 3])

    def evaluate(self, a):
        t_traj = self.obj.trolly(a)
        vib = self.obj.RK4(self.c_traj[:,0], self.c_traj[:,1], t_traj[:,2])
        # return error1(vib)
        # return error2(vib)
        error, _, _ = error3(self.obj, t_traj, self.c_traj, vib)
        return error

    def update_pos(self, a, va):
        return a + va

    def update_vel(self, a, va, p, g, w_=0.730, p_=2.05):
        ro1 = random.uniform(0, 1)
        ro2 = random.uniform(0, 1)
        return w_*(va + p_*ro1*(p-a) + p_*ro2*(g-a))

    def cal(self):
        print('Initializing variables')
        parti_count = 100  # 粒子の数 : 100
        a_min, a_max = -0.5, 0.5

        param_count = 5
        pos = np.array([[random.uniform(a_min, a_max) for i in range(param_count)] for j in range(parti_count)])
        vel = np.array([[0.0 for i in range(param_count)] for j in range(parti_count)])

        p_best_pos = copy.deepcopy(pos)
        p_best_scores = [self.evaluate(p) for p in pos]

        best_parti = np.argmin(p_best_scores)
        g_best_pos = p_best_pos[best_parti]

        loop_count = 200  # 制限時間 : 200
        print('Start calculation')
        for t in range(loop_count):
            print('{}/{}'.format(t+1, loop_count))
            for n in tqdm(range(parti_count)):
                # n番目のx, y, vx, vyを定義
                a = pos[n]
                va = vel[n]
                p = p_best_pos[n]

                # 粒子の位置の更新
                new_a = self.update_pos(a, va)
                pos[n] = new_a

                # 粒子の速度の更新
                new_va = self.update_vel(a, va, p, g_best_pos)
                vel[n] = new_va

                # 評価値を求め、パーソナルベストの更新
                score = self.evaluate(new_a)
                if score < p_best_scores[n]:
                    p_best_scores[n] = score
                    p_best_pos[n] = new_a

            # グローバルベストの更新
            best_parti = np.argmin(p_best_scores)
            g_best_pos = p_best_pos[best_parti]

            t_traj_gbest = self.obj.trolly(g_best_pos)
            vib_gbest = self.obj.RK4(self.c_traj[:,0], self.c_traj[:,1], t_traj_gbest[:,2])
            self.progress[t, :] = list(error3(self.obj, t_traj_gbest, self.c_traj, vib_gbest))

        print(f'係数: {g_best_pos}')

        return g_best_pos

    
if __name__ == '__main__':
    args = sys.argv

    start = time.time()
   
    for exeCount in range(10):
        optim = PSO()
        np.savetxt('./results/pso/val3/cond1/pso_params_{}'.format(exeCount), optim.cal(), delimiter=',')
        np.savetxt('./results/pso/val3/cond1/progress_report_{}'.format(exeCount), optim.progress, delimiter=',')

    elapsed_time = time.time() - start
    print(f'Elapsed Time : {elapsed_time} [sec]')
