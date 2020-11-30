import os
import warnings
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore", category=FutureWarning)

proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']

properties = np.zeros(6)


def extract_features(openPath):
    label = []
    c = -1
    i = 0
    train_set = []
    kind = ["B2F", "B3F", "C2F", "C3F", "C4F"]

    for root, dirs, files in os.walk(openPath):

        for file in files:
            file_path = os.path.join(root, file)

            # raw_image = cv2.imread(file_path, 0)
            #
            # image_resize = cv2.resize(raw_image, (256, 500))
            #
            # cv2.imwrite(file_path, image_resize)

            image = cv2.imread(file_path, 0)
            glcmMatrix = (greycomatrix(image, [1], [0], levels=2 ** 8))
            for j in range(0, len(proList)):
                properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))
            features = np.array(
                [properties[0], properties[1], properties[2], properties[3], properties[4], properties[5], kind[c]])
            train_set.append(features)
        c += 1
    return train_set


if __name__ == '__main__':
    train_path = r'C:\Classify\train_831'
    test_path = r'C:\Classify\test_351'
    train_set = extract_features(train_path)
    test_set = extract_features(test_path)

    X = np.asarray(np.asmatrix(train_set))
    X_on_test = np.asarray(np.asmatrix(test_set))

    Y = X[:, -1]  # 于Y中存储标签
    X = X[:, 0:-1]  # 于X中存储特征

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=17)
    y_on_test = X_on_test[:, -1]
    X_on_test = X_on_test[:, 0:-1]
    saved = []
    models = []
    results = []
    names = []

    models.append(('CART', DecisionTreeClassifier(max_depth=4)))
    models.append(('SVM', SVC(C=1.0)))
    models.append(('GBC', GradientBoostingClassifier(n_estimators=105)))
    models.append(("KNN", KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='kd_tree')))

    for name, model in models:
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        names.append(name)
        results.append(score * 100)
        saved.append(model)
        print(name, score * 100)

    i = np.argmax(results)
    print("\nmax on train set:", names[i], '  ', results[i])

    GBC = GradientBoostingClassifier(n_estimators=105)
    GBC.fit(X_train, y_train)
    score = GBC.score(X_on_test, y_on_test)
    print('\nGBC on test set accuracy =', score * 100)
