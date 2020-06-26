import numpy as np
from collections import Counter

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    # Helper method que recibe un sample
    def _predict(self, x):
        # Calcular distancias
        distances = [euclidean_distance(x, x_train)
                     for x_train in self.X_train]

        # Conocer los k samples mas cercanos, para obtener sus labels

        # Va de menor a mayor
        # Sortear distancias (se devuelve los indices) y tomar las k mas pequeñas
        k_indices = np.argsort(distances)[:self.k]

        # Obtener las labels de esos samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Obtener la clase más comun, de las k labels obtenidas
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]  # Se retorna la clase


def euclidean_distance(x1, x2):  # x1 y x2 son feature vectors
    return np.sqrt(np.sum((x1 - x2) ** 2))


def main():
                            # red,      green,      azul
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,

                                                    random_state=1234)
    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = np.sum(predictions == y_test) / len(y_test)
    print("\n (k = 3) Accuracy del modelo con el test dataset: ", acc * 100, "%")

    clf = KNN(k=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = np.sum(predictions == y_test) / len(y_test)
    print("\n (k = 5) Accuracy del modelo con el test dataset: ", acc * 100, "%")




if __name__ == "__main__":
    main()
