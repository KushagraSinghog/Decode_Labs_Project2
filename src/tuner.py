import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def evaluate_k_values(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    max_k: int = 25
) -> Tuple[List[int], List[float]]:
    """
    Evaluates different values of K (from 1 to max_k) using 5-fold cross-validation on training data.
    This prevents test data leakage and helps avoid overfitting (K=1) and underfitting (large K).
    
    Args:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training labels.
        max_k (int): Maximum value of K to evaluate.
        
    Returns:
        k_values (List[int]): Evaluated values of K.
        error_rates (List[float]): Average cross-validation error rate (1 - accuracy) for each K.
    """
    k_values = list(range(1, max_k + 1))
    error_rates = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        # Using 5-fold cross-validation
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        # Error rate is 1 - accuracy
        error_rate = 1.0 - np.mean(scores)
        error_rates.append(error_rate)
        
    return k_values, error_rates

def find_optimal_k(k_values: List[int], error_rates: List[float]) -> int:
    """
    Automatically selects the optimal K value based on the lowest error rate.
    To avoid overfitting (Slide 12: K=1 is noisy), we prefer K > 1.
    We also prefer odd values of K to prevent ties in voting.
    
    Args:
        k_values (List[int]): List of evaluated K values.
        error_rates (List[float]): Error rates for each K.
        
    Returns:
        optimal_k (int): The chosen optimal K.
    """
    min_error = min(error_rates)
    
    # Get all K values that achieve the minimum error
    best_ks = [k for k, err in zip(k_values, error_rates) if err == min_error]
    
    # Filter out K=1 if possible, to avoid overfitting/noise
    filtered_ks = [k for k in best_ks if k > 1]
    if not filtered_ks:
        filtered_ks = best_ks
        
    # Prefer odd K values to avoid voting ties
    odd_ks = [k for k in filtered_ks if k % 2 != 0]
    if odd_ks:
        return odd_ks[0]
        
    return filtered_ks[0]

def plot_elbow_curve(
    k_values: List[int], 
    error_rates: List[float], 
    optimal_k: int,
    save_path: str = None
) -> None:
    """
    Generates and optionally saves the Elbow Curve plot (Error Rate vs. K Value)
    to visually tune the KNN model (Slide 12).
    
    Args:
        k_values (List[int]): List of K values.
        error_rates (List[float]): List of error rates.
        optimal_k (int): The selected optimal K to highlight on the plot.
        save_path (str): Filepath to save the figure (e.g. 'elbow_plot.png').
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the elbow curve
    plt.plot(k_values, error_rates, color='#1e3d59', linestyle='dashed', marker='o',
             markerfacecolor='#ff6e40', markersize=8, linewidth=2, label='CV Error Rate')
    
    # Highlight the optimal K
    optimal_idx = k_values.index(optimal_k)
    plt.plot(optimal_k, error_rates[optimal_idx], marker='o', markersize=14, 
             markeredgecolor='red', markerfacecolor='none', markeredgewidth=3, 
             label=f'Optimal K ({optimal_k})')
    
    # Styling - Premium Aesthetics
    plt.title('Tuning the Engine: Choosing "K" (Elbow Method)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('K Value (Number of Neighbors)', fontsize=12, labelpad=10)
    plt.ylabel('Error Rate (1 - CV Accuracy)', fontsize=12, labelpad=10)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)
    
    # Add explanatory text inside the plot
    plt.text(0.05, 0.05, 
             'Low K: High Variance (Overfitting)\nHigh K: High Bias (Underfitting)', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', alpha=0.8, edgecolor='#cccccc'))
    
    plt.tight_layout()
    
    if save_path:
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
