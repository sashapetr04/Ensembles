import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
import time

def mean_squared_error(y1, y2):
    return np.mean((y1 - y2) ** 2)

class RandomForestMSE:
    def __init__(
        self, n_estimators=50, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.models = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """

        if self.feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        history = {'error': np.zeros(self.n_estimators), 'time': np.zeros(self.n_estimators)}

        for i in range(self.n_estimators):
            random_state = np.random.randint(1, 1000)

            bootstrap_indices = np.random.choice(X.shape[0], X.shape[0])
            model = DecisionTreeRegressor(max_depth=self.max_depth,
                                          max_features=self.feature_subsample_size,
                                          random_state=random_state,
                                          **self.trees_parameters)

            start = time.time()
            model.fit(X[bootstrap_indices], y[bootstrap_indices])
            end = time.time()

            self.models.append(model)

            y_pred = self.predict(X_val)
            history['error'][i] = np.mean((y_pred - y_val) ** 2) ** 0.5
            history['time'][i] = end - start

        return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        predictions = np.zeros(X.shape[0])
        for i, model in enumerate(self.models):
            predictions += model.predict(X)
        return predictions / len(self.models)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators=50, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.models = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        if self.feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        history = np.zeros(self.n_estimators)

        predictions = np.zeros_like(y)
        self.alphas = np.zeros(self.n_estimators)

        history = {'error': np.zeros(self.n_estimators), 'time': np.zeros(self.n_estimators)}

        for i in range(self.n_estimators):
            random_state = np.random.randint(1, 1000)

            model = DecisionTreeRegressor(max_depth=self.max_depth,
                                          max_features=self.feature_subsample_size,
                                          random_state=random_state,
                                          **self.trees_parameters)
            start = time.time()
            model.fit(X, y - predictions)
            self.models.append(model)

            alpha = minimize_scalar(lambda x: mean_squared_error(y, predictions + x * model.predict(X))).x
            self.alphas[i] = alpha
            predictions += self.learning_rate * self.alphas[i] * model.predict(X)

            y_pred = self.predict(X_val)
            end = time.time()

            history['error'][i] = np.mean((y_pred - y_val) ** 2) ** 0.5
            history['time'][i] = end - start

        return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        predictions = np.zeros(X.shape[0])

        for i, model in enumerate(self.models):
            predictions += self.learning_rate * self.alphas[i] * model.predict(X)

        return predictions
