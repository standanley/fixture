import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1.8, 1.8, 21)
y1 = x
y2 = np.maximum(-1, np.minimum(1, x))
y3 = np.tanh(x)


plt.plot(x, y1, '-*')
plt.plot(x, y2, '-x')
plt.plot(x, y3, '-+')
plt.legend(['y = x', 'y = clamp(x, -1, 1)', 'y = tanh(x)'])
plt.grid()
plt.show()