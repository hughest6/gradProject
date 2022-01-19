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
import pandas as pd
from data_processing import *


def model_comparison():
    dat = pd.read_csv('comparison_metrics.csv')
    false_alarm = []
    true_detect = []
    snr_list = []
    for i in range(0, int(len(dat) / 10)):
        at_snr = dat.loc[i * 10:(i * 10) + 9]
        snr = at_snr['SNR'].mean()
        me = at_snr['false_alarm'].mean()
        me_p = at_snr['true_pred'].mean()
        false_alarm.append(me)
        true_detect.append(me_p)
        snr_list.append(snr)
        print(at_snr)
        print("false alarm average " + str(me))
        print("true detection average " + str(me_p))
        print(snr_list)

    print(false_alarm)
    print(true_detect)
    print(snr_list)

    plt.subplot(2, 1, 1)
    plt.plot(snr_list, true_detect)
    plt.title('Correct Detections')
    plt.xlabel('SNR (dB)')
    plt.ylabel('% Correct Detections')
    plt.subplot(2, 1, 2)
    plt.plot(snr_list, false_alarm)
    plt.title('False Alarms')
    plt.xlabel('SNR (dB)')
    plt.ylabel('% False Alarm')
    plt.subplots_adjust(hspace=.5)
    plt.savefig('SNR_Performance.png')
    plt.show()


def gen_comparison():
    pass


if __name__ == '__main__':
    freqs = range(int(1E9), int(2E9), int(1E8))
    freq = [1E8, 2E8, 3E8]
    theta = [-20, -10, 0, 10, 20]

    t0 = time.time()
    d, raw = DataHandler.generate_table(5000, 1, 6, freqs, theta, 1000)
    d2, raw2 = DataHandler.generate_table(5000, 1, 6, freqs, theta, 20)
    m = t_flow(raw, len(freqs), len(theta))
    m.save("saved_model\mymodel")

    #m = tf.keras.models.load_model("saved_model\mymodel")
    # cyl_scene = Scene(freqs, theta)
    # tri_scene = Scene(freqs, theta)
    # plt_scene = Scene(freqs, theta)
    # cyl_ref = CylinderReflector(cyl_area=2, cyl_length=2)
    # cyl_ref.randomize()
    # tri_ref = TrihedralReflector(area=2)
    # tri_ref.randomize()
    # plt_ref = PlateReflector(plate_area=2.1)
    # plt_ref.randomize()
    #
    # cyl_scene.add_reflector(cyl_ref, 1)
    # tri_scene.add_reflector(tri_ref, 1)
    # plt_scene.add_reflector(plt_ref, 1)
    #
    # snr_range = range(-6, 30, 2)
    # print(len(snr_range))
    # tests_per_snr = 2
    # trials = 200

    # with open('comparison_metrics.csv', mode='w') as csv_file:
    #     fieldnames = ['SNR', 'test_num', 'trials', 'false_alarm', 'true_pred']
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for snr in snr_range:
    #         print("hererererer")
    #         print(snr)
    #         cyl_scene.set_snr(snr)
    #         tri_scene.set_snr(snr)
    #         plt_scene.set_snr(snr)
    #         for c in range(0, tests_per_snr):
    #             fa, p = model_probabilities(m, trials, trials, plt_scene, (cyl_scene, tri_scene))
    #             writer.writerow({'SNR': snr, 'test_num': c, 'trials': trials, 'false_alarm': fa, 'true_pred': p})
    #



    #model_comparison()

#   DataHandler.write_file(d, 'train_file_no_noise')
#   DataHandler.write_file(d2, 'test_file_m20db')

#   file_data = r'C:\Users\tyler\PycharmProjects\gradProject\gradProject\Data\\'
#   filetype = '.csv'
#   file_loc1 = file_data + 'train_file_no_noise' + filetype
#   file_loc2 = file_data + 'test_file_m20db' + filetype
#   training_set = data_prep(file_loc1)
#   testing_set = data_prep(file_loc2)

#  # t_predict(m, raw2)
#   t1 = time.time()
#   t2 = time.time()
#   standard_tree = train_tree(training_set[0], training_set[1])
#   loc = file_data+'decision_tree_conf_matrix.png'
#   forrest = random_forest(training_set[0], training_set[1])
#   extra_t = extra_trees(training_set[0], training_set[1])
#   predictor(standard_tree, testing_set[2], testing_set[3], 'standard_tree 0dB')
#   predictor(forrest, testing_set[2], testing_set[3], 'random_forest 0dB')
#   predictor(extra_t, testing_set[2], testing_set[3], 'extra_trees 0dB')
#   t3 = time.time()
#   #mlp = mlp_classifier(prepped[0], prepped[1])
#   #predictor(mlp, prepped[2], prepped[3])

#   print('File preparation execution completed in: ' + str(t1-t0) + ' seconds')
#   print('Model training completed in: ' + str(t3-t2) + ' seconds')
