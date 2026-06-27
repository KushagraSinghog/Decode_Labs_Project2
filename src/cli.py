import os
import sys

# Allow running this file directly as a script
if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
from src.pipeline import run_pipeline
from src.model import IrisKNNClassifier

BUILD_DIR = 'build'
MODEL_PATH = os.path.join(BUILD_DIR, 'model.joblib')
SCALER_PATH = os.path.join(BUILD_DIR, 'scaler.joblib')

# Class labels mapping
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*60)
    print("       DECODELABS PROJECT 2: DATA CLASSIFICATION USING AI")
    print("               Iris Species KNN Predictor CLI")
    print("="*60)

def train_pipeline_interactive():
    print("\n--- Model Training & Optimization Pipeline ---")
    print(f"This will train the KNN model, tune 'K', and output artifacts to '{BUILD_DIR}/'.")
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Pipeline cancelled.")
        return

    try:
        results = run_pipeline(output_dir=BUILD_DIR)
        print("\n[SUCCESS] Model trained successfully!")
        print(f"Optimal K selected: {results['optimal_k']}")
        print(f"Model accuracy on test set: {results['metrics']['accuracy']:.4f}")
        input("\nPress Enter to return to main menu...")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during pipeline run: {e}")
        input("\nPress Enter to return to main menu...")

def predict_interactive():
    print("\n--- Real-Time Species Inference ---")
    
    # Verify model and scaler artifacts exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[WARNING] Trained model and scaler files not found in the 'build/' folder.")
        print("Please train the model first by selecting Option 1 in the main menu.")
        input("\nPress Enter to return...")
        return
        
    try:
        # Load artifacts
        print("Loading trained model and scaler...")
        model = IrisKNNClassifier.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
        
        # Get user inputs
        print("\nPlease enter the physical dimensions of the Iris flower:")
        
        sepal_length = get_float_input("1. Sepal Length (cm) [Typical range: 4.3 - 7.9]: ", 0.1, 20.0)
        sepal_width  = get_float_input("2. Sepal Width (cm)  [Typical range: 2.0 - 4.4]: ", 0.1, 20.0)
        petal_length = get_float_input("3. Petal Length (cm) [Typical range: 1.0 - 6.9]: ", 0.1, 20.0)
        petal_width  = get_float_input("4. Petal Width (cm)  [Typical range: 0.1 - 2.5]: ", 0.1, 20.0)
        
        # Structure features into array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale features using the loaded scaler
        features_scaled = scaler.transform(features)
        
        # Perform prediction and get class probabilities
        pred_class_idx = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        predicted_species = TARGET_NAMES[pred_class_idx].capitalize()
        confidence = probabilities[pred_class_idx] * 100
        
        print("\n" + "-"*40)
        print("               INFERENCE RESULTS")
        print("-"*40)
        print(f"Predicted Species:  \033[1;32m{predicted_species}\033[0m")
        print(f"Confidence Level:   \033[1;34m{confidence:.2f}%\033[0m")
        print("\nProbability Distribution:")
        for idx, name in enumerate(TARGET_NAMES):
            prob = probabilities[idx] * 100
            print(f" - {name.capitalize():<10}: {prob:.2f}%")
        print("-"*40)
        
        input("\nPress Enter to perform another prediction or return...")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred during inference: {e}")
        input("\nPress Enter to return...")

def get_float_input(prompt: str, min_val: float, max_val: float) -> float:
    while True:
        try:
            val_str = input(prompt).strip()
            val = float(val_str)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"[INPUT ERROR] Value must be between {min_val} and {max_val}. Please try again.")
        except ValueError:
            print("[INPUT ERROR] Invalid numeric entry. Please enter a float/decimal number.")

def show_tuner_details():
    print("\n--- Tuning Parameter K Detail ---")
    print("This runs the grid search tuner on K-neighbors and displays the cross-validation error.")
    confirm = input("Run parameter sweep? (y/n): ").strip().lower()
    if confirm != 'y':
        return
        
    from src.data_loader import load_iris_data, split_dataset, scale_features
    from src.tuner import evaluate_k_values
    
    try:
        X, y, _ = load_iris_data()
        X_train, _, y_train, _ = split_dataset(X, y)
        X_train_scaled, _, _ = scale_features(X_train, X_train) # Dummy test split
        
        print("\nEvaluating K values (1 to 25) via 5-fold cross-validation...")
        k_values, error_rates = evaluate_k_values(X_train_scaled, y_train, max_k=25)
        
        print("\n" + "-"*35)
        print(f" {'K-Neighbors':<12} | {'CV Error Rate':<15}")
        print("-"*35)
        for k, err in zip(k_values, error_rates):
            highlight = " <-- Min Error" if err == min(error_rates) else ""
            print(f" {k:<12} | {err:<15.4f}{highlight}")
        print("-"*35)
        
        input("\nPress Enter to return to main menu...")
    except Exception as e:
        print(f"\n[ERROR] Tuning failed: {e}")
        input("\nPress Enter to return...")

def main():
    while True:
        clear_screen()
        print_header()
        print(" 1. Run Complete Training & Preprocessing Pipeline")
        print(" 2. Predict Iris Species (Interactive Real-Time Input)")
        print(" 3. View K-Parameter Hyperparameter Tuning Details")
        print(" 4. Exit CLI")
        print("="*60)
        
        choice = input("Enter option (1-4): ").strip()
        
        if choice == '1':
            train_pipeline_interactive()
        elif choice == '2':
            predict_interactive()
        elif choice == '3':
            show_tuner_details()
        elif choice == '4':
            print("\nThank you for using the Iris KNN Predictor. Exiting...")
            sys.exit(0)
        else:
            print("\n[INPUT ERROR] Invalid selection. Please enter a number between 1 and 4.")
            import time
            time.sleep(1.5)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCLI terminated by user. Exiting...")
        sys.exit(0)
