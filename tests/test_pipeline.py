import unittest
import numpy as np
import pandas as pd
from src.data_loader import load_iris_data, split_dataset, scale_features
from src.model import IrisKNNClassifier
from src.tuner import evaluate_k_values, find_optimal_k
from src.metrics import compute_classification_metrics

class TestIrisClassificationPipeline(unittest.TestCase):
    
    def setUp(self):
        # Setup dummy data for simple test operations
        self.X_dummy = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9] * 3,
            'sepal_width':  [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1] * 3,
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5] * 3,
            'petal_width':  [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1] * 3
        })
        self.y_dummy = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 2)
        self.target_names = ['setosa', 'versicolor', 'virginica']

    def test_load_iris_data(self):
        X, y, target_names = load_iris_data()
        self.assertEqual(X.shape, (150, 4))
        self.assertEqual(len(y), 150)
        self.assertEqual(len(target_names), 3)
        self.assertListEqual(target_names, ['setosa', 'versicolor', 'virginica'])

    def test_split_dataset(self):
        X, y, target_names = load_iris_data()
        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)
        
        # Check standard 80/20 train-test ratio (150 * 0.8 = 120, 150 * 0.2 = 30)
        self.assertEqual(X_train.shape, (120, 4))
        self.assertEqual(X_test.shape, (30, 4))
        self.assertEqual(len(y_train), 120)
        self.assertEqual(len(y_test), 30)

    def test_scale_features(self):
        X_train, X_test, _, _ = split_dataset(self.X_dummy, self.y_dummy, test_size=0.2, random_state=42)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Verify the training features are scaled correctly (mean ~ 0, std ~ 1)
        mean_scaled = np.mean(X_train_scaled, axis=0)
        std_scaled = np.std(X_train_scaled, axis=0)
        
        np.testing.assert_array_almost_equal(mean_scaled, np.zeros(4), decimal=5)
        np.testing.assert_array_almost_equal(std_scaled, np.ones(4), decimal=5)
        
        # Ensure transformation dimensions match inputs
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)

    def test_model_training_and_prediction(self):
        # Scale
        X_train_scaled, X_test_scaled, _ = scale_features(self.X_dummy, self.X_dummy)
        
        # Fit model
        model = IrisKNNClassifier(n_neighbors=3)
        model.fit(X_train_scaled, self.y_dummy)
        
        # Predict
        preds = model.predict(X_test_scaled)
        self.assertEqual(len(preds), len(self.y_dummy))
        
        # Predict probabilities
        probs = model.predict_proba(X_test_scaled)
        self.assertEqual(probs.shape, (len(self.y_dummy), 3))
        # Probabilities sum to 1.0 along class axis
        np.testing.assert_array_almost_equal(np.sum(probs, axis=1), np.ones(len(self.y_dummy)), decimal=5)

    def test_tuner_k_selection(self):
        X_train_scaled, _, _ = scale_features(self.X_dummy, self.X_dummy)
        k_values, error_rates = evaluate_k_values(X_train_scaled, self.y_dummy, max_k=5)
        
        self.assertEqual(len(k_values), 5)
        self.assertEqual(len(error_rates), 5)
        
        # Check selection logic
        optimal_k = find_optimal_k(k_values, error_rates)
        self.assertTrue(1 <= optimal_k <= 5)
        
    def test_compute_metrics(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        
        metrics = compute_classification_metrics(y_true, y_pred, self.target_names)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('macro_precision', metrics)
        self.assertIn('macro_recall', metrics)
        self.assertIn('macro_f1', metrics)
        
        # Accuracy check: 4 correct predictions out of 6 -> 4/6 = 0.6667
        self.assertAlmostEqual(metrics['accuracy'], 4/6, places=4)
        
        # Structure check
        self.assertIn('setosa', metrics['class_metrics'])
        self.assertIn('versicolor', metrics['class_metrics'])
        self.assertIn('virginica', metrics['class_metrics'])

if __name__ == '__main__':
    unittest.main()
