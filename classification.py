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
    disp.ax_.set_title('Confusion Matrix' + str(net.__class__) + '\n SNR: -30dB')
    print('Confusion Matrix')
    print(disp.confusion_matrix)
    dirc = os.path.dirname(__file__)
    filename = os.path.join(dirc, 'Data', name+'.jpg')
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
