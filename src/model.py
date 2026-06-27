import joblib
import numpy as np
from typing import Self
from sklearn.neighbors import KNeighborsClassifier

class IrisKNNClassifier:
    """
    A wrapper around the scikit-learn KNeighborsClassifier for Iris species classification.
    Provides standard scikit-learn workflow (Instantiate -> Fit -> Predict) as outlined in Slide 13,
    along with helper utility methods for save/load serialization.
    """
    def __init__(
        self, 
        n_neighbors: int = 5, 
        weights: str = 'uniform', 
        metric: str = 'minkowski', 
        p: int = 2
    ):
        """
        Initializes the K-Nearest Neighbors Classifier wrapper.
        
        Args:
            n_neighbors (int): Number of neighbors to use (default is 5).
            weights (str): Weight function used in prediction ('uniform' or 'distance').
            metric (str): Distance metric to use for the tree (default 'minkowski').
            p (int): Power parameter for the Minkowski metric (default 2, which equates to Euclidean).
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fits the KNN model to training data (Slide 13: "Fit (Memorize the map)").
        
        Args:
            X (np.ndarray): Preprocessed training features.
            y (np.ndarray): Target labels.
            
        Returns:
            self: The fitted classifier instance.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the input features (Slide 13: "Predict (Apply logic)").
        
        Args:
            X (np.ndarray): Preprocessed test/inference features.
            
        Returns:
            predictions (np.ndarray): Predicted class labels.
        """
        if not hasattr(self.model, "classes_"):
            raise ValueError("Model is not fitted yet. Call 'fit' before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probability estimates for each class for input features.
        
        Args:
            X (np.ndarray): Preprocessed test/inference features.
            
        Returns:
            probabilities (np.ndarray): Array of shape (n_samples, n_classes) with class probabilities.
        """
        if not hasattr(self.model, "classes_"):
            raise ValueError("Model is not fitted yet. Call 'fit' before predicting probabilities.")
        return self.model.predict_proba(X)

    def save(self, filepath: str) -> None:
        """
        Saves the fitted model object to a file for later deployment or inference.
        
        Args:
            filepath (str): Absolute or relative file path to save the model.
        """
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'IrisKNNClassifier':
        """
        Loads a saved IrisKNNClassifier object from file.
        
        Args:
            filepath (str): File path to load the model from.
            
        Returns:
            model (IrisKNNClassifier): Loaded model instance.
        """
        loaded_obj = joblib.load(filepath)
        if not isinstance(loaded_obj, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return loaded_obj
