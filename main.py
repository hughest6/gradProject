from reflectors import *
from rcs_stats import *
from data_processing import *
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

freqs = range(int(1E9), int(2E9), int(1E8))
freq = [1E8, 2E8, 3E8]
theta = [-20, -10, 0, 10, 20]

scene3 = Scene(freqs, theta)
scene3.add_random_reflectors(5)

d = DataHandler.generate_table(15, 1, 6, freqs, theta)
DataHandler.write_file(d, 'testfile')
