import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_iris_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Loads the Iris dataset from scikit-learn.
    
    Returns:
        X (pd.DataFrame): Dataframe containing the 4 features (sepal/petal lengths & widths).
        y (pd.Series): Series containing target labels (0, 1, 2).
        target_names (List[str]): Names of the target species corresponding to labels.
    """
    iris = load_iris()
    # Create DataFrame for features
    X = pd.DataFrame(iris.data, columns=[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
    ])
    y = pd.Series(iris.target, name='species')
    target_names = list(iris.target_names)
    
    return X, y, target_names

def split_dataset(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and test sets.
    Standard requirement is 80% train, 20% test with shuffle to remove order bias (Slide 10).
    
    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target labels.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # shuffle=True is default in train_test_split, but we pass it explicitly to be clear
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True, 
        stratify=y  # Stratification ensures balanced class distributions in splits (Slide 8, 14)
    )

def scale_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Applies StandardScaler to scale features such that mean=0 and variance=1 (Slide 9).
    The scaler is fit on the training data and applied to both train and test data.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        
    Returns:
        X_train_scaled (np.ndarray): Standardized training features.
        X_test_scaled (np.ndarray): Standardized testing features.
        scaler (StandardScaler): The fitted scaler object (for inference scaling).
    """
    scaler = StandardScaler()
    # Fit scaler only on training data to avoid data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test data using the fitted scaler parameters
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
