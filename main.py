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
import os
import time
import tensorflow as tf
import csv


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
#print(len(freqs))
freq = [1E8, 2E8, 3E8]
theta = [-20, -10, 0, 10, 20]

tri = CylinderReflector()
tri.randomize()
scene3 = Scene(freqs, theta)
scene3.add_reflector(tri, 0)
scene3.print_reflectors()
#print(scene3.scene_statistics())

t0 = time.time()
#d, raw = DataHandler.generate_table(5000, 1, 6, freqs, theta)
#m = t_flow(raw, len(freqs), len(theta))
#m.save("saved_model\mymodel")

m = tf.keras.models.load_model("saved_model\mymodel")
cyl_scene = Scene(freqs, theta)
tri_scene = Scene(freqs, theta)
plt_scene = Scene(freqs, theta)
cyl_ref = CylinderReflector(cyl_area=2, cyl_length=2)
cyl_ref.randomize()
tri_ref = TrihedralReflector(area=2)
tri_ref.randomize()
plt_ref = PlateReflector(plate_area=2.1)
plt_ref.randomize()

cyl_scene.add_reflector(cyl_ref, 1)
tri_scene.add_reflector(tri_ref, 1)
plt_scene.add_reflector(plt_ref, 1)

snr_range = range(-50, 10, 5)
print(len(snr_range))
tests_per_snr = 10
trials = 1000

with open('comparison_metrics.csv', mode='w') as csv_file:
    fieldnames = ['SNR', 'test_num', 'trials', 'false_alarm', 'true_pred']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for snr in snr_range:
        print(snr)
        cyl_scene.set_snr(snr)
        tri_scene.set_snr(snr)
        plt_scene.set_snr(snr)
        for c in range(0, tests_per_snr):
            fa, p = model_probabilities(m, trials, trials, plt_scene, (cyl_scene, tri_scene))
            writer.writerow({'SNR': snr, 'test_num': c, 'trials': trials, 'false_alarm': fa, 'true_pred': p})



fa, p = model_probabilities(m, 10, 10, plt_scene, (cyl_scene, tri_scene))




#DataHandler.write_file(d, 'testfile')

#file_data = r'C:\Users\tyler\PycharmProjects\gradProject\gradProject\Data\\'
#filetype = '.csv'
#file_loc = file_data + 'testfile' + filetype
#prepped = data_prep(file_loc)

#kmeans_cluster(prepped[0])
#tup_size = [5,5]
#print(random.weibull(1, tup_size))
#print(random.rayleigh(1, tup_size))
#print(random.normal(0,1,tup_size))
t1 = time.time()

t2 = time.time()
#standard_tree = train_tree(prepped[0], prepped[1])
#loc = file_data+'decision_tree_conf_matrix.png'
#forrest = random_forest(prepped[0], prepped[1])
#extra_t = extra_trees(prepped[0], prepped[1])
#predictor(standard_tree, prepped[2], prepped[3], 'standard_tree -40dB')
#predictor(forrest, prepped[2], prepped[3], 'random_forest -40dB')
#predictor(extra_t, prepped[2], prepped[3], 'extra_trees -40dB')
t3 = time.time()
#mlp = mlp_classifier(prepped[0], prepped[1])
#predictor(mlp, prepped[2], prepped[3])

print('File preparation execution completed in: ' + str(t1-t0) + ' seconds')
print('Model training completed in: ' + str(t3-t2) + ' seconds')
