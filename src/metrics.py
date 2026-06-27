import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_names: List[str]
) -> Dict[str, any]:
    """
    Computes professional evaluation metrics including Accuracy, Precision, Recall, and F1-Score (Slide 14, 16).
    Handles multiclass metrics by calculating weighted and macro averages.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        target_names (List[str]): List of class names.
        
    Returns:
        metrics_dict (Dict): Dictionary containing computed metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, f1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(target_names))
    )
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Class-specific metrics dict
    class_metrics = {}
    for i, name in enumerate(target_names):
        class_metrics[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
        
    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'class_metrics': class_metrics
    }

def print_metrics_summary(metrics: Dict[str, any]) -> None:
    """
    Prints a beautiful ASCII summary table of the computed metrics in the console.
    """
    print("\n" + "="*50)
    print("           MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:   {metrics['macro_f1']:.4f}")
    print("-"*50)
    print(f"{'Species':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-"*50)
    for class_name, cls_metrics in metrics['class_metrics'].items():
        print(f"{class_name:<15} | {cls_metrics['precision']:<10.4f} | {cls_metrics['recall']:<10.4f} | {cls_metrics['f1_score']:<10.4f} | {cls_metrics['support']:<8}")
    print("="*50 + "\n")

def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_names: List[str],
    save_path: str = None
) -> None:
    """
    Generates and saves a Confusion Matrix heatmap (Slide 15).
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        target_names (List[str]): List of class names for labels.
        save_path (str): Filepath to save the plot image.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    plt.figure(figsize=(8, 6))
    
    # Using a professional, high-contrast color map (blues/purples)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True,
                annot_kws={'size': 14, 'weight': 'bold'}, linewidths=1.5, linecolor='#ffffff')
    
    # Styling
    plt.title('The Diagnostic Tool: Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Species', fontsize=12, labelpad=10)
    plt.ylabel('True Species', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def get_text_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> str:
    """
    Returns a textual representation of the confusion matrix for the command line.
    """
    cm = confusion_matrix(y_true, y_pred)
    max_len = max(len(name) for name in target_names)
    
    lines = []
    lines.append("\nCONFUSION MATRIX:")
    header = f"{' ' * (max_len + 3)}" + "".join(f"{name:>{max_len + 2}}" for name in target_names)
    lines.append(header)
    lines.append("-" * len(header))
    
    for i, true_name in enumerate(target_names):
        row_str = f"{true_name:<{max_len}} | "
        for j in range(len(target_names)):
            row_str += f"{cm[i, j]:>{max_len + 2}}"
        lines.append(row_str)
    
    return "\n".join(lines) + "\n"
