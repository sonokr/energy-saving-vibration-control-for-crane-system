import configparser
import copy
import random
import sys
import time

import numpy as np
from numba import jit
from platypus import NSGAII, Problem, Real
from scipy import integrate
from tqdm import tqdm

from common_functions import energy, find_inflection_point, load_conditions
from simulator import Crane

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


obj = Crane()
c_traj = obj.cable()


def schaffer(a):
    print(a)

    t_traj = obj.trolly(a)
    vib = obj.RK4(c_traj[:,0], c_traj[:,1], t_traj[:,2])
    error = np.amax(abs(vib[0, Nte+1:]))

    f = obj.force(vib[0, :], vib[1, :], 
                  t_traj[0:2*Nrk+1:2, :], c_traj[0:2*Nrk+1:2, :])
    x = copy.deepcopy(t_traj[:2*Nte+1:2, 0])
    ene = energy(f, x)

    return [error, ene]


def cal(args, exeCount):
    start = time.time()
    param_count = 5  # パラメーター数

    problem = Problem(param_count, 2)  # 最適化パラメータの数, 目的関数の数
    problem.types[:] = Real(-2.0, 2.0)  # パラメータの範囲
    problem.function = schaffer
    algorithm = NSGAII(problem)
    algorithm.run(5000)  # 反復回数

    print('{:-^63}'.format('-'))

    # データ整理
    # params: 係数a
    # f1s   : 残留振動 [deg]
    # f2s   : エネルギー
    params = np.empty([100, param_count])
    f1s = np.empty([100])
    f2s = np.empty([100])
    for i, solution in enumerate(algorithm.result):
        result = tuple(solution.variables \
                     + solution.objectives[:])

        params[i, :] = result[:param_count][:]
        f1s[i] = 180*result[param_count]/np.pi
        f2s[i] = result[param_count+1]

    # 残留振動が最小になるaの値を表示
    index = np.argmin(f1s)
    print('\n*** 残留振動が最小の時の各値 ***')
    print('残留振動[deg]\t{}'.format(f1s[index]))
    print('エネルギー[J]\t{}'.format(f2s[index]))
    print('係数a\t\t{}'.format(params[index, :]))

    np.savetxt('./results/nsga2/{}/nsga2_params_{}.csv'.format(args[1], exeCount),
               params[index, :], delimiter=',')

    # 経過時間
    print(f'\nelapsed time: {time.time()-start}')

    # 係数a, 残留振動, エネルギーをCSVファイルに書き出す
    data = np.empty([100, param_count+2])
    data[:, 0:param_count] = params
    data[:, param_count] = f1s
    data[:, param_count+1] = f2s
    np.savetxt('./results/nsga2/{}/nsga2_data_{}.csv'.format(args[1], exeCount),
               data, delimiter=',')


if __name__ == '__main__':
    args = sys.argv

    for exeCount in range(10):
        cal(args, exeCount)
