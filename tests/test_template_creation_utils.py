import fixture
import math
import random


def test_invert_function(plot=False):
    def fun(x):
        return math.atan(10*(x-.5))/math.pi+.5

    N = 50
    xs = [i/N for i in range(N+1)]

    #random.seed(0)
    ys_ideal = [fun(x) for x in xs]
    ys = [y_ideal + 0.3*(random.random()-0.5) for y_ideal in ys_ideal]

    inv = fixture.template_creation_utils.invert_function(xs, ys)

    ys_test = [i/(50*N) for i in range(-10, 50*N+11)]
    xs_test = inv(ys_test)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(xs, ys, '*')
        plt.plot(inv.y, inv.x)
        plt.legend(['Measured points', 'Approximation'])
        #plt.plot(xs_test, ys_test)
        plt.grid()
        plt.show()

    for i in range(len(xs_test) - 1):
        assert xs_test[i+1] >= xs_test[i]

if __name__ == '__main__':
    test_invert_function(plot=True)