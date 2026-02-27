"""
Model evaluation utilities for news popularity prediction.

Provides functions to compute classification and regression metrics,
generate confusion matrices, and create evaluation reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_regression_metrics(y_true, y_pred, prefix=""):
    """
    Compute regression performance metrics.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth target values.
    y_pred : array-like, shape (n_samples,)
        Predicted target values.
    prefix : str, default=""
        Prefix for metric names (e.g., "train_" or "test_").
    
    Returns
    -------
    metrics : dict
        Dictionary of metric names and values.
    """
    metrics = {
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}r2": r2_score(y_true, y_pred),
        f"{prefix}mse": mean_squared_error(y_true, y_pred)
    }
    return metrics


def compute_classification_metrics(y_true, y_pred, y_pred_proba=None, prefix=""):
    """
    Compute classification performance metrics.
    
    Note: For binary classification tasks. Assumes positive class = 1.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth binary labels (0 or 1).
    y_pred : array-like, shape (n_samples,)
        Predicted binary labels (0 or 1).
    y_pred_proba : array-like, shape (n_samples,), optional
        Predicted probabilities for positive class. Required for ROC-AUC.
    prefix : str, default=""
        Prefix for metric names.
    
    Returns
    -------
    metrics : dict
        Dictionary of metric names and values.
    """
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        try:
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            # Handle case where only one class present in y_true
            print(f"Warning: Could not compute ROC-AUC. {e}")
            metrics[f"{prefix}roc_auc"] = np.nan
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, 
                         title="Confusion Matrix", figsize=(8, 6), 
                         save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Class labels for display. If None, uses unique values from y_true.
    normalize : bool, default=False
        If True, normalize confusion matrix by row (true label).
    title : str, default="Confusion Matrix"
        Plot title.
    figsize : tuple, default=(8, 6)
        Figure size.
    save_path : str, optional
        If provided, save figure to this path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    cm : np.ndarray
        Confusion matrix array.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    if labels is None:
        labels = np.unique(y_true)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig, ax, cm


def generate_classification_report(y_true, y_pred, y_pred_proba=None, 
                                   target_names=None, output_dict=False):
    """
    Generate comprehensive classification report.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_pred_proba : array-like, optional
        Predicted probabilities for ROC-AUC calculation.
    target_names : list, optional
        Display names for classes.
    output_dict : bool, default=False
        If True, return as dictionary instead of string.
    
    Returns
    -------
    report : str or dict
        Classification report with per-class metrics.
    """
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )
    
    if output_dict and y_pred_proba is not None:
        try:
            report['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            report['roc_auc'] = np.nan
    
    return report


def evaluate_model(y_true, y_pred, y_pred_proba=None, task='regression', 
                   prefix="", verbose=True):
    """
    Unified evaluation function for both regression and classification.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predictions (continuous for regression, labels for classification).
    y_pred_proba : array-like, optional
        Predicted probabilities (classification only).
    task : str, default='regression'
        Either 'regression' or 'classification'.
    prefix : str, default=""
        Prefix for metric names (e.g., "train_", "test_").
    verbose : bool, default=True
        If True, print metrics to console.
    
    Returns
    -------
    metrics : dict
        All computed metrics.
    """
    if task == 'regression':
        metrics = compute_regression_metrics(y_true, y_pred, prefix=prefix)
    elif task == 'classification':
        metrics = compute_classification_metrics(
            y_true, y_pred, y_pred_proba=y_pred_proba, prefix=prefix
        )
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"{prefix.upper() if prefix else 'EVALUATION'} METRICS")
        print(f"{'='*50}")
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        print(f"{'='*50}\n")
    
    return metrics


# def save_metrics_to_file(metrics, filepath="results/model_metrics.json"):
#     """
#     Save metrics dictionary to JSON file.
    
#     Parameters
#     ----------
#     metrics : dict
#         Metrics to save.
#     filepath : str
#         Output filepath.
#     """
#     import json
#     import os
    
#     os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
#     with open(filepath, 'w') as f:
#         json.dump(metrics, f, indent=2)
    
#     print(f"Metrics saved to {filepath}")
