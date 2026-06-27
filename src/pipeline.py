import os
import sys
import joblib

# Allow running this file directly as a script
if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from src.data_loader import load_iris_data, split_dataset, scale_features
from src.model import IrisKNNClassifier
from src.tuner import evaluate_k_values, find_optimal_k, plot_elbow_curve
from src.metrics import (
    compute_classification_metrics, 
    print_metrics_summary, 
    plot_confusion_matrix,
    get_text_confusion_matrix
)

def run_pipeline(output_dir: str = 'build') -> Dict[str, Any]:
    """
    Executes the complete machine learning pipeline:
    1. Loads Iris dataset.
    2. Performs stratified 80/20 train-test split.
    3. Fits StandardScaler on train set and scales both splits.
    4. Evaluates K (1-25) using 5-fold cross-validation.
    5. Determines optimal K and saves the Elbow Plot.
    6. Trains the final KNN model with optimal K.
    7. Evaluates predictions using Accuracy, Precision, Recall, and F1-Score.
    8. Generates and saves the Confusion Matrix heatmap.
    9. Serializes the final model and scaler for deployment.
    
    Args:
        output_dir (str): Directory where artifacts (plots, model, scaler) will be saved.
        
    Returns:
        pipeline_results (Dict): Paths of saved artifacts and training metadata.
    """
    print("Starting DecodeLabs Project 2 AI Pipeline...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print("[1/9] Loading Iris dataset...")
    X, y, target_names = load_iris_data()
    
    # 2. Split Data (Slide 10)
    print("[2/9] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 3. Scale Features (Slide 9)
    print("[3/9] Applying StandardScaler scaling...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 4. Tune K Parameter (Slide 12)
    print("[4/9] Tuning KNN hyperparameter 'K' (range: 1-25)...")
    k_values, error_rates = evaluate_k_values(X_train_scaled, y_train, max_k=25)
    
    # 5. Determine Optimal K & Save Elbow Plot (Slide 12)
    optimal_k = find_optimal_k(k_values, error_rates)
    print(f"      -> Optimal K identified: {optimal_k}")
    
    elbow_plot_path = os.path.join(output_dir, 'elbow_plot.png')
    plot_elbow_curve(k_values, error_rates, optimal_k, save_path=elbow_plot_path)
    print(f"      -> Saved Elbow Curve visualization to: {elbow_plot_path}")
    
    # 6. Train final model (Slide 13)
    print(f"[6/9] Training final model with K={optimal_k}...")
    model_wrapper = IrisKNNClassifier(n_neighbors=optimal_k)
    model_wrapper.fit(X_train_scaled, y_train)
    
    # 7. Predict on Test Set (Slide 13)
    print("[7/9] Running predictions on scaled test set...")
    y_pred = model_wrapper.predict(X_test_scaled)
    
    # 8. Evaluate & Compute Metrics (Slide 14, 15, 16)
    print("[8/9] Calculating evaluation metrics and generating Confusion Matrix...")
    metrics = compute_classification_metrics(y_test, y_pred, target_names)
    print_metrics_summary(metrics)
    
    # Text confusion matrix for quick console inspection
    txt_cm = get_text_confusion_matrix(y_test, y_pred, target_names)
    print(txt_cm)
    
    # Save confusion matrix plot
    cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, target_names, save_path=cm_plot_path)
    print(f"      -> Saved Confusion Matrix heatmap to: {cm_plot_path}")
    
    # 9. Save serialized artifacts
    print("[9/9] Serializing model and scaler for deployment...")
    model_path = os.path.join(output_dir, 'model.joblib')
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    
    model_wrapper.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"      -> Saved Model wrapper to: {model_path}")
    print(f"      -> Saved Scaler object to: {scaler_path}")
    print("\nPipeline execution completed successfully!")
    
    return {
        'optimal_k': optimal_k,
        'metrics': metrics,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'elbow_plot_path': elbow_plot_path,
        'cm_plot_path': cm_plot_path,
        'target_names': target_names
    }

if __name__ == '__main__':
    # Can run this file directly to test the pipeline
    run_pipeline()
