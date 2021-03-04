import numpy as np
import matplotlib.pyplot as plt

freq = 1e9
amplitude = .5




def plot(should_be_1, alpha, beta, gamma):
    def f(v, s):
        return sum([
            should_be_1 * v,
            alpha * v * s,
            beta * s ** 2,
            gamma * s
        ])

    N = 1000
    t_total = 10 / freq
    f_sample = t_total / N

    #x = np.arange(0, 5 / freq,  5 / freq / N)
    x = np.linspace(0, t_total, N+1)[:N]
    y = amplitude * np.sin(x * 2 * np.pi * freq)
    slope = np.gradient(y, x[1] - x[0])
    sampled = [f(v, s) for v, s in zip(y, slope)]

    plt.plot(x, y)
    plt.plot(x, sampled)
    plt.show()

    freqs = np.linspace(0, 1 / f_sample, N+1)[:N]
    fft = np.abs(np.fft.fft(sampled)) / N

    assert len(fft) % 2 == 0
    # say freqs = 1/8*[0 1 2 3 4 5 6 7] * f_sample
    # we want the result to be
    #                            [0 1 2 3]
    #                          +   [7 6 5 4]
    unalias = np.concatenate((fft[:len(fft)//2], [0]))
    alias = np.concatenate(([0], fft[-1 : len(fft)//2-1 : -1]))
    fft_folded = unalias + alias
    freqs_folded = freqs[0:len(freqs)//2+1]


    #fft = fft[:len(fft)//2]
    #plt.loglog(freqs, fft)
    plt.loglog(freqs_folded, fft_folded)
    plt.show()


# sample_out = should_be_1*value + slope * effective_delay
# effective_delay = alpha*value + beta*slope + gamma*1

# = value*should_be_1 + slope*value*alpha + slope^2*beta + slope*gamma

#should_be_1 = 0.996
#alpha = -1.46e-12 * 1e9
#beta = 8.88e-25 * 1e18
#gamma = -2.37e-13 * 1e9


should_be_1 = 1
alpha = 0
beta = 0
gamma = .1 / freq

#plot(should_be_1, alpha, beta, gamma)

should_be_1 = 1
alpha = .3 / freq
beta = 0
gamma = 0

#plot(should_be_1, alpha, beta, gamma)

should_be_1 = 1
alpha = 0
beta = .03 / freq**2
gamma = 0

#plot(should_be_1, alpha, beta, gamma)

should_be_1 = 0.996
alpha = -1.46e-12 #* 1e9
beta = 8.88e-25 #* 1e18
gamma = -2.37e-13 #* 1e9


plot(should_be_1, alpha, beta, gamma)