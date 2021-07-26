from reflectors import *
from rcs_stats import *
from data_processing import *
from scene import *
from scipy.stats import weibull_min
import numpy as np
import matplotlib.pyplot as plt
from classification import *
from sklearn import tree
from numpy import random


c = 10.79
#fig, ax = plt.subplots(1, 1)
#x = np.linspace(weibull_min.ppf(0.01, c),
#                weibull_min.ppf(0.99, c), 100)
#plt.plot(x, weibull_min.pdf(x, c),
#         'r-', lw=5, alpha=0.6, label='weibull_min pdf')

if __name__ == '__main__':
    rv = weibull_min.rvs(c, size=10)
    #print(rv)

freqs = range(int(1E9), int(2E9), int(1E8))
freq = [1E8, 2E8, 3E8]
theta = [-20, -10, 0, 10, 20]

scene3 = Scene(freqs, theta)
scene3.add_random_reflectors(1)
scene3.scene_statistics()
scene3.print_reflectors()

#d = DataHandler.generate_table(5000, 1, 6, freq, theta)
#DataHandler.write_file(d, 'testfile')

file_data = r'C:\Users\tyler\PycharmProjects\gradProject\gradProject\Data\\'
filetype = '.csv'
file_loc = file_data + 'testfile' + filetype
prepped = data_prep(file_loc)

#kmeans_cluster(prepped[0])
#tup_size = [5,5]
#print(random.weibull(1, tup_size))
#print(random.rayleigh(1, tup_size))
#print(random.normal(0,1,tup_size))

standard_tree = train_tree(prepped[0], prepped[1])
loc = file_data+'decision_tree_conf_matrix.png'
forrest = random_forest(prepped[0], prepped[1])
extra_t = extra_trees(prepped[0], prepped[1])
predictor(standard_tree, prepped[2], prepped[3])
predictor(forrest, prepped[2], prepped[3])
predictor(extra_t, prepped[2], prepped[3])

#mlp = mlp_classifier(prepped[0], prepped[1])
#predictor(mlp, prepped[2], prepped[3])

