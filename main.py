from rcsFunctions import *
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt


c = 10.79
fig, ax = plt.subplots(1, 1)
x = np.linspace(weibull_min.ppf(0.01, c),
                weibull_min.ppf(0.99, c), 100)
plt.plot(x, weibull_min.pdf(x, c),
         'r-', lw=5, alpha=0.6, label='weibull_min pdf')


if __name__ == '__main__':
    x = plate_rcs(4, 40, 10000)
    print(x)
    rv = weibull_min.rvs(c, size=10)
    print(rv)


