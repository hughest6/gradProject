import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import os
import tensorflow as tf
import numpy as np


dct = {
        'trihedral_reflector': 0,
        'cylinder_reflector': 1,
        'plate_reflector': 2
    }


def data_prep(file_loc):
    scaler = StandardScaler()
    ref_proc = pd.read_csv(file_loc)
    train_set, test_set = train_test_split(ref_proc, test_size=0.2, random_state=42)

    train_y = train_set['obj_type']
    train_x = train_set.drop('obj_type', axis=1)
    test_y = test_set['obj_type']
    test_x = test_set.drop('obj_type', axis=1)

    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    train_y = [dct[k] for k in train_y]
    test_y = [dct[k] for k in test_y]

    return [train_x, train_y, test_x, test_y]


def count_predictions(test, predicted):
    """
        Given a set of dicotomials for both expected and predicted outputs, returns a pair of values counting
        correct and wrong predictions.
    """
    correct = 0
    wrong = 0

    for i in list(zip(test, predicted)):
        if i[0] == i[1]:
            correct += 1
        else:
            wrong += 1
    return correct, wrong


def train_dec_regressor(X,y):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X, y)
    return tree_reg


def train_tree(X, y):
    clftod = tree.DecisionTreeClassifier(max_depth=3, random_state=1234)  # Decision Tree classifier
    clftod = clftod.fit(X, y)
    return clftod


def kmeans_cluster(X):
    kmeans = KMeans(n_clusters=3).fit(X)
    kmeans_y = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=kmeans_y)
    plt.title('Target Clustering Data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    #plt.show()
    print(silhouette_score(X, kmeans_y))


def mlp_classifier(X, y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
    clf.fit(X, y)
    return clf


def random_forest(X, y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split = 2, random_state = 0, n_jobs=-1)
    clf.fit(X, y)
    return clf


def extra_trees(X,y):
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0, n_jobs=-1)
    clf.fit(X, y)
    return clf


def predictor(net, test_x, test_y, name):
    y_pred = net.predict(test_x)
    correct, wrong = count_predictions(test_y, y_pred)
    print(net.__class__)
    print("We obtained {} correct ({}%) predictions against {} wrong one".format(correct,round(100*correct/(correct+wrong),2),wrong))

    disp = plot_confusion_matrix(net, test_x, test_y, display_labels=dct.keys(),cmap=plt.cm.Blues)
    disp.ax_.set_title('Confusion Matrix' + str(net.__class__) + '\n SNR: 0dB')
    print('Confusion Matrix')
    print(disp.confusion_matrix)
    dirc = os.path.dirname(__file__)
    filename = os.path.join(dirc, 'Data', name+'.jpg')
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def t_flow(raw_data, num_freqs, num_thetas):
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42)
    print("test set size:")
    print(len(test_set))
    y_train = []
    x_train = []
    for obj in train_set:
        y_train.append(obj[0]-1)
        x_train.append(obj[1])
    y_test = []
    x_test = []
    for obj in test_set:
        y_test.append(obj[0]-1)
        x_test.append(obj[1])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    x_test = np.asarray(x_test)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

    #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    BATCH_SIZE = 1
    SHUFFLE_BUFFER_SIZE = 100

    #train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    #test_dataset = test_dataset.batch(BATCH_SIZE)
    #inshape=(1, num_thetas, num_freqs)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(num_thetas, num_freqs)),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Flatten(input_shape=(num_thetas, num_freqs)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=30, batch_size=1)
    y_pred = model.predict(x_test)
    print(tf.math.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), num_classes=3))

    return model


def t_predict(model, raw_data):
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42)
    print("test set size:")
    print(len(test_set))
    y_train = []
    x_train = []
    for obj in train_set:
        y_train.append(obj[0] - 1)
        x_train.append(obj[1])
    y_test = []
    x_test = []
    for obj in test_set:
        y_test.append(obj[0] - 1)
        x_test.append(obj[1])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    x_test = np.asarray(x_test)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

    y_pred = model.predict(x_test)
    correct, wrong = count_predictions(np.argmax(y_pred, 1), np.argmax(y_test, 1))
    print("We obtained {} correct ({}%) predictions against {} wrong one".format(correct, round(
        100 * correct / (correct + wrong), 2), wrong))

    print(tf.math.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), num_classes=3))


def model_probabilities(model, fa_trials, d_trials, true_ref, false_ref):
    print("calculating probabilities")
    num_false_ref = len(false_ref)
    inc = 0
    false_alarms = 0
    for i in range(0, fa_trials):
        if inc == num_false_ref-1:
            inc = 0
        else:
            inc = inc + 1
        sc = false_ref[inc]
        t = tf.expand_dims(sc.scene_rcs(), axis=0)
        pred = model.predict(t)

        m_ind = np.argmax(pred)
        if m_ind == 0:
            false_alarms = false_alarms + 1

    detections = 0
    for i in range(0, d_trials):
        sc = true_ref
        t = tf.expand_dims(sc.scene_rcs(), axis=0)
        pred = model.predict(t)

        m_ind = np.argmax(pred)
        if m_ind == 0:
            detections = detections + 1

    fa_p = false_alarms/fa_trials * 100
    tar_p = detections/d_trials * 100
    print("FA % :" + str(fa_p))
    print("Detections %: " + str(tar_p))
    return fa_p, tar_p
