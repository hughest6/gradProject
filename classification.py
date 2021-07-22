import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

dct = {
        'trihedral_reflector': 0,
        'cylinder_reflector': 1,
        'plate_reflector': 2
    }


def data_prep(file_loc):
    ref_proc = pd.read_csv(file_loc)
    train_set, test_set = train_test_split(ref_proc, test_size=0.2, random_state=42)
    train_y = train_set['obj_type']
    train_x = train_set.drop('obj_type', axis=1)
    train_y = [dct[k] for k in train_y]
    train_x = normalize(train_x, axis=1)

    test_y = test_set['obj_type']
    test_x = test_set.drop('obj_type', axis=1)
    test_y = [dct[k] for k in test_y]
    test_x = normalize(test_x, axis=1)

    return [train_x, train_y, test_x, test_y]


def train_dec_regressor(X,y):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X, y)
    return tree_reg


def train_tree(X, y):
    clftod = tree.DecisionTreeClassifier(max_depth=3, random_state=1234)  # Decision Tree classifier
    clftod = clftod.fit(X, y)
    return clftod


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


def kmeans_cluster(X):
    kmeans = KMeans(n_clusters=3).fit(X)
    kmeans_y = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=kmeans_y)
    plt.title('Target Clustering Data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    #plt.show()
    print(silhouette_score(X, kmeans_y))