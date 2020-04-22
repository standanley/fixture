import pickle
import fixture
from scipy.optimize import minimize
import numpy as np

f = open('temp_impulse_response.pickle', 'rb')
(ts, ys) = pickle.load(f)
#fixture.template_creation_utils.plot(t, y)
#exit()

#print(type(ts), type(ys))


t0_guess = ts[np.argmax(ys)]
unit = 1 / t0_guess
print(t0_guess)
h_guess = max(ys)
w_guess = .9 / (.5 * h_guess)
h2_guess = h_guess
w2_guess = .1 / (0.5 * h2_guess)

def piecewise(t, x):
    (t0, w, h, w2, h2) = x
    t0, w, h, w2, h2 = t0/unit, w/unit, h*unit, w2 / unit, h2*unit
    if t < t0 - w:
        return 0
    elif t < t0:
        return h*(t - (t0-w))/w
    elif t < t0 + w2:
        return h2 * ((t0+w2) - t)/w2
    else:
        return 0

def fun(x, ts, ys):
    #retval = sum((y-piecewise(t, x))**2 for t, y in zip(ts, ys))
    s = 0
    for t, y in zip(ts, ys):
        a = (y-piecewise(t, x))
        b = a**2
        s += b
    retval = s
    return retval

x0 = np.array([t0_guess * unit, w_guess * unit, h_guess / unit, w2_guess * unit, h2_guess / unit])

#print(piecewise(-.5e-6, x0))
#exit()


result = minimize(fun, x0, args = (ts, ys))
x_min = result.x
print('x_min is', x_min)
print('x0 is', x0)

area1 = .5 * x_min[1] * x_min[2]
area2 = .5 * x_min[3] * x_min[4]
print(area1, area2, area1+area2)

area0 = 0
for i in range(len(ts)-1):
    area0 += ys[i] * (ts[i+1] - ts[i])
    area0 += .5 * (ys[i+1] - ys[i]) * (ts[i+1] - ts[i])
print(area0)
    
import matplotlib.pyplot as plt
plt.plot(ts, ys)
plt.plot(ts, [piecewise(t, x0) for t in ts])
plt.plot(ts, [piecewise(t, x_min) for t in ts])
plt.grid()
plt.show()
