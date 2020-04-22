import pickle
import fixture
from scipy.optimize import minimize
import numpy as np

f = open('temp_impulse_response.pickle', 'rb')
(ts, ys) = pickle.load(f)
#fixture.template_creation_utils.plot(t, y)
#exit()

#print(type(ts), type(ys))

area0 = 0
for i in range(len(ts)-1):
    area0 += ys[i] * (ts[i+1] - ts[i])
    #area0 += .5 * (ys[i+1] - ys[i]) * (ts[i+1] - ts[i])
print(area0)

t0_guess = ts[np.argmax(ys)]
unit = 1 / t0_guess
print(t0_guess)
h_guess = max(ys)
w_guess = area0 * .9 / (.5 * h_guess)
h2_guess = h_guess

def piecewise(t, x):
    (t0, w, h, h2) = x
    t0, w, h, h2 = t0/unit, w/unit, h*unit, h2*unit
    area = .5 * w * h
    area2 = area0 - area
    w2 = area2 / (.5 * h2)
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

#t0_guess = ts[np.argmax(ys)]
#print(t0_guess)
#h_guess = max(ys)
#w_guess = .9 / (.5 * h_guess)
#h2_guess = h_guess
x0 = np.array([t0_guess * unit, w_guess * unit, h_guess / unit, h2_guess / unit])

# this x0 based on optimiation without total area constraint
# 0.4495888,   
#x0 = [ 2.43532231, 17.74239167,  0.08205216,  0.06967387]

#print(piecewise(-.5e-6, x0))
#exit()


result = minimize(fun, x0, args = (ts, ys))
x_min = result.x
print('x_min is', x_min)
print('x0 is', x0)

print('trapezoidal area', area0)
area1 = .5 * x_min[1] * x_min[2]
area2 = .5 * x_min[3] * x_min[4]
print('fit area', area1, area2, area1+area2)


    
import matplotlib.pyplot as plt
plt.plot(ts, ys)
plt.plot(ts, [piecewise(t, x0) for t in ts])
plt.plot(ts, [piecewise(t, x_min) for t in ts])
plt.grid()
plt.show()
