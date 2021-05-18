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


plate1 = PlateReflector(4, 1.3, 1E8)
cylinder1 = CylinderReflector(3, 3, 1E6)

print(cylinder1.rcs())
print(plate1.rcs())

print(plate1.location)

scene1 = Scene()
scene1.add_reflector(plate1, 20)
scene1.add_reflector(cylinder1, 50)
scene1.print_reflectors()
print(scene1.reflector_count)

scene1.remove_reflector(plate1)
scene1.print_reflectors()
print(scene1.reflector_count)

scene1.clear_all_reflectors()
scene1.print_reflectors()

scene2 = Scene()
scene2.add_random_reflectors(5)
scene2.print_reflectors()
