from reflectors import *
from scene import *
from scipy.stats import weibull_min
import numpy as np
import matplotlib.pyplot as plt


c = 10.79
fig, ax = plt.subplots(1, 1)
x = np.linspace(weibull_min.ppf(0.01, c),
                weibull_min.ppf(0.99, c), 100)
plt.plot(x, weibull_min.pdf(x, c),
         'r-', lw=5, alpha=0.6, label='weibull_min pdf')


if __name__ == '__main__':
    rv = weibull_min.rvs(c, size=10)
    print(rv)


scene3 = Scene()
scene3.add_random_reflectors(5)
theta = [-20, -10, 0, 10, 20]
freq = [1E8, 2E8, 3E8]
rcs = scene3.scene_rcs(theta, freq)
print(rcs)
scene3.print_reflectors()


