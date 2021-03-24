import numpy as np
import matplotlib.pyplot as plt

# This file is my attempt to analyze results where the std dev
# of the measurements is linear in one of the input variables.
# My conclusion is that it's not too straightforward, the best
# approach is probably optimization_slow, and in 1D maybe chunks
# can work.
# And in general you probably want at least a couple hundred samples?

PLOT = True

NOISE_SLOPE = 0.3
NOISE_INTERCEPT = 1.7

def est_ys(x, linear):
    m, b = linear
    est = x**2*m**2 + x*2*m*b + b**2
    return est


def quadratic_no_constraint(x, y):
    fit = np.polyfit(x, y, 2)
    y_est = x ** 2 * fit[0] + x * fit[1] + fit[2]
    return fit, y_est

def quadratic_then_constraint(x, y):
    fit, _ = quadratic_no_constraint(x, y)
    noise_slope = fit[0]**.5
    noise_intercept = fit[1] / (2*noise_slope)
    est = est_ys(x, (noise_slope, noise_intercept))
    return [noise_slope, noise_intercept], est

def ideal_cheating(x, _):
    fit = [NOISE_SLOPE, NOISE_INTERCEPT]
    est = est_ys(x, fit)
    return [NOISE_SLOPE, NOISE_INTERCEPT], est

def cleverness_bad(x, y):
    # I want the best fit y = ax^2 + bx + c, but actually
    # it needs to be constrained to y = (d**2)x^2 + (2*d*e)x + (e**2)
    # I thought this was clever: y = (dx + e)**2
    #                      sqrt(y) = dx + e
    # and now it's a linear regression
    # But the answer it gives me is trash ... even when the quadratic fit
    # gives me a,b,c such that you can find a d,e that work, the linear
    # regression does not find those d,e at all.
    # There is a difference in the way different points are weighted, and
    # it doesn't work because the average of the squareroots in not the
    # squareroot of the average (in fact, it's like somebody designed the
    # distribution so those quantities would be extremelky different)
    fit = np.polyfit(x, y**.5, 1)
    est = est_ys(x, fit)
    return fit, est

def optimization_slow(x, y):
    from scipy.optimize import minimize
    def fun(params):
        d, e = params
        return sum((res2 - ((d ** 2) * x ** 2 + (2 * d * e) * x + e ** 2)) ** 2)

    #print(fun([0, 1]))
    #print(fun([.4, 1.5]))
    #print(fun([0.3, 1.7]))
    opt = minimize(fun, [0.4, 1.5])
    fit = opt.x
    est = est_ys(x, fit)
    return fit, est

def chunks(x, y, num=10):
    new_xs = []
    avgs = []
    N = len(x)
    #size = len(x)//num
    for i in range(num):
        xs = x[(i*N)//num:((i+1)*N)//num]
        ys = y[(i*N)//num:((i+1)*N)//num]
        new_x = sum(xs) / len(xs)
        avg = sum(ys) / len(ys)
        new_xs.append(new_x)
        avgs.append(avg)
    new_ys = [avg**.5 for avg in avgs]
    fit = np.polyfit(new_xs, new_ys, 1)
    est = est_ys(x, fit)
    return fit, est





N = 100
noise = np.random.normal(size=(N,))
x = np.random.uniform(size=(N,)) * 100
testx = np.array(x[:100])
x.sort()
testx.sort()

noise = noise * (x * NOISE_SLOPE + NOISE_INTERCEPT)
signal = 1.3 * x + 5 
y = signal + noise

fit = np.polyfit(x, y, 1)
y_est = x * fit[0] + fit[1]

if False or PLOT:
    plt.plot(x, y, '*')
    plt.plot(x, y_est)
    plt.grid()
    plt.show()


res = y - y_est
res2 = res**2

tests = [
    ('ideal', ideal_cheating),
    #'quadratic, no constraint', quadratic_no_constraint
    ('estimate from quadratic', quadratic_then_constraint),
    ('cleverness, bad', cleverness_bad),
    ('optimization', optimization_slow),
    ('Chunks', chunks)
]

fit_ests = [f(x, res2) for _, f in tests]


def error(a, b):
    assert len(a) == len(b)
    return sum((a0-b0)**2 for a0, b0 in zip(a, b)) / len(a)

ideal_est = fit_ests[0][1]
errors = [error(ideal_est, est) for _, est in fit_ests]
for (name, _), error in zip(tests, errors):
    print(name, error)


if PLOT:
    plt.title('Residuals')
    plt.plot(x, res2, '*')
    for _, est in fit_ests:
        plt.plot(x, est)

    names = ['data'] + [n for n, _ in tests]

    plt.legend(names)
    plt.show()

    exit()

    plt.title('Residuals')
    plt.plot(x, res2, '*')
    #plt.plot(x, res2_est**.5)
    #plt.plot(x, res2_est2**.5)

    #plt.plot(testx, testy1**.5)
    #plt.plot(testx, testy2**.5)
    #plt.plot(testx, testy3**.5)
    plt.show()

# we should find that the variance at x is (x*.3+0.7)**2=.09x^2*.42x+.49
#print(noise_slope, noise_intercept)

#if not PLOT:
#    plt.plot(testx, testy1)
#    plt.plot(testx, testy2)
#    plt.plot(testx, testy3)
#    plt.show()



