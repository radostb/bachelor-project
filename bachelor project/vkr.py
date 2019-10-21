from mpmath import besseli as iv, besselj as jv, besselk as kv
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from numpy import sqrt, log, pi
from numpy.linalg import inv
from mpmath import mp, invertlaplace
from math import pi as mpi
import pandas as pd
import time
import matplotlib.pyplot as plt
%matplotlib inline

class second(object):
    def __init__(self, a, h, mu, r0, pc):
        self.a = a
        self.h = h
        self.mu = mu
        self.r0 = r0
        self.pc = pc
    def calc_B(self, p, pi):
        sp = sqrt(p)
        numenator = (pi - self.pc)/p
        denominator = kv(0, self.r0 * sp / self.a)
        return numenator / denominator
    def diff(self, p, pi):
        B = self.calc_B(p,pi)
        K1 = kv(1, self.r0*sqrt(p)/self.a)
        return (-sqrt(p)/self.a)*B*K1
    def calc_dH(self, p, pi, h, k):
        D = self.diff(p, pi)
        return 2*mpi*h*self.r0*k*D/self.mu

    def calc(self, t, pi, k):
        h = self.h
        fp = lambda p: self.calc_dH(p, pi, h, k)
        return invertlaplace(fp, t, method='stehfest', degree=10)
    @staticmethod
    def run(self, t, pi, k):
        return self.calc(t, pi, k)*3600*24


well=second((10**(-13)/(10**(-3)*0.2*10**(-4)/101325))**0.5, 10, 10**(-3),0.1, 250*101325)
N = 3
T = 360
P_initial = 250*101325
p = np.array([100*101325, 20*101325, 10*101325, 200*101325])
k = 1e-13
x = np.arange(10, N*T*10, 10)
y = np.arange(0.002778, 3, 0.002778)
z = [well.calc(j, p[0], k)*3600*24 for j in x]
for i in range(T,N*T-1):
    z[i]-=well.calc((i-T+1)*10, P_initial - p[1] + p[0], k)*3600*24
    if i in range(2*T, N*T-1):
        z[i]-=well.calc((i-2*T+1)*10, P_initial - p[2] + p[1], k)*3600*24

plt.plot(y, z)
plt.xlabel('t, hour')
plt.ylabel('q, m3/d')

#Алгоритм Левенберга-Марквардта:
def norma4L2(a):
    s = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            s += a[i, j]**2
    return s**(1/2)
#@jit
def norma4infinity(a):
    return max(abs(a))
#@jit
def f(x, model, t, p):
    z = [well.calc(j, p[0], x[0, 0])*3600*24 for j in t]
    for i in range(T,N*T-1):
        z[i]-=well.calc((i-T+1)*10, P_initial - p[1] + p[0], x[0, 0])*3600*24
        if i in range(2*T, N*T-1):
            z[i]-=well.calc((i-2*T+1)*10, P_initial - p[2] + p[1], x[0, 0])*3600*24
            #if i in range(3*T, N*T-1):
            # z[i]-=well.calc((i-3*T+1)*10, P_initial - p[3] + p[2], x[0, 0])*3600*24
    z = np.array(z)
    f_ = z - model
    f_ = np.array(f_).reshape((len(f_)),1)
    return f_

def F(x, model, t, p):
    f_ = f(x, model, t, p)

    F_ = 0
    for i in range(len(t)):
        F_ += f_[i]**2
    return F_/2

def J(x, t, p):
    J0 = [(well.calc(i, p[0], x[0, 0] + x[0, 0]/100) - well.calc(i, p[0], x[0, 0] - x[0, 0]/100)) / (2*(x[0, 0] / 100)) for i in t]
    J_ = np.zeros((len(t), len(x)))
    for i in range(len(t)):
        J_[i, 0] = J0[i]
    return J_
def levmar(model):
    eps1 = 1e-8
    eps2 = 1e-8
    tau = 1e-3
    N = 3
    T = 360
    it = 0
    max_it = 100
    nu = 2
    k = 2e-13
    P_initial = 250*101325
    p = np.array([100*101325, 20*101325, 10*101325, 200*101325])

    t = np.arange(10, N*T*10, 10)
    x = np.zeros((1, 1))
    x[0, 0] = k
    J_ = J(x, t, p)
    A = np.dot(np.transpose(J_), J_)
    I = np.eye(len(A))
    g = np.dot(np.transpose(J_), f(x, model, t, p))
    if norma4infinity(g) <= eps1:
        found = True
    else:
        found = False
        mu = tau * A.max()
    while (not found) and it < max_it:
        it += 1
        print(".................")
        C = np.array(A+mu*I, dtype=float)
        h1m = np.dot(inv(C), (-g))
        if norma4L2(h1m) <= eps2*(norma4L2(x)+eps2):
            found = True
        else:
            x_new = x + h1m

            ro = (F(x, model, t, p)-F(x_new, model, t, p)) / (0.5*np.dot(np.transpose(h1m),(mu*h1m - g)))
            ro = ro[0, 0]
            if ro > 0:
                x = x_new
                print("x: ")
                print(x)
                print("h1m: ")
                print(h1m)
                J_ = J(x, t, p)
                A = np.dot(np.transpose(J_), J_)
                g = np.dot(np.transpose(J_), f(x, model, t, p))
                if norma4infinity(g) <= eps1:
                    found = True
                    mu = mu*max(1/3, 1-(2*ro-1)**3)
                    nu = 2
                #else:
                # found = False
                else:
                    mu = mu*nu
                    nu = 2*nu
                    print("mu: ")
                    print(mu)
                    
    print(".................")
    print("number of iterations: %.0f" %it)
    return x


x = np.zeros((1, 1))
timer1 = time.time()
x = levmar(z)
timer2 = time.time()
print('elapsed time: %.2f sec' %(timer2-timer1))

#Проверка на устойчивость:
import random
l = len(z)
z_02 = np.zeros(l)
z_04 = np.zeros(l)
z_06 = np.zeros(l)
z_08 = np.zeros(l)
z_1 = np.zeros(l)
for i in range(l):
    z_02[i] = random.uniform(z[i] - 0.002*z[i], z[i] + 0.002*z[i])
    z_04[i] = random.uniform(z[i] - 0.004*z[i], z[i] + 0.004*z[i])
    z_06[i] = random.uniform(z[i] - 0.006*z[i], z[i] + 0.006*z[i])
    z_08[i] = random.uniform(z[i] - 0.008*z[i], z[i] + 0.008*z[i])
    z_1[i] = random.uniform(z[i] - 0.01*z[i], z[i] + 0.01*z[i])

x_02 = np.zeros((1, 1))
timer1 = time.time()
x_02 = levmar(z_02)
timer2 = time.time()
print('elapsed time: %.2f sec' %(timer2-timer1))
x_04 = np.zeros((1, 1))

timer1 = time.time()
x_04 = levmar(4_02)
timer2 = time.time()
print('elapsed time: %.2f sec' %(timer2-timer1))
x_06 = np.zeros((1, 1))
timer1 = time.time()
x_06 = levmar(z_06)
timer2 = time.time()
print('elapsed time: %.2f sec' %(timer2-timer1))
x_08 = np.zeros((1, 1))
timer1 = time.time()
x_08 = levmar(z_08)
timer2 = time.time()
print('elapsed time: %.2f sec' %(timer2-timer1))
x_1 = np.zeros((1, 1))
timer1 = time.time()
x_1 = levmar(z_1)
timer2 = time.time()
print('elapsed time: %.2f sec' %(timer2-timer1))

#Вывод результатов:
plt.plot(y, z_1, label='С шумом')
plt.plot(y, z, label='Без шума')
plt.xlabel('t, hour')
plt.ylabel('q, m3/d')
plt.legend(loc='upper right')

x = np.arange(10, N*T*10, 10)
k = 9.9894073255948378e-14
z_model = [well.calc(j, p[0], k)*3600*24 for j in x]
for i in range(T,N*T-1):
    z_model[i]-=well.calc((i-T+1)*10, P_initial - p[1] + p[0], k)*3600*24
    if i in range(2*T, N*T-1):
        z_model[i]-=well.calc((i-2*T+1)*10, P_initial - p[2] + p[1], k)*3600*24

plt.plot(y, z_model, label='С полученным k')
plt.plot(y, z, label='С исходным k')
plt.xlabel('t, hour')
plt.ylabel('q, m3/d')
plt.legend(loc='upper right')