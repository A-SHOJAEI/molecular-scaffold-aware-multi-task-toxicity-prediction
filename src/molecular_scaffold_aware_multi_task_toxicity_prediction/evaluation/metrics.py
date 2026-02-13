"""Evaluation metrics and analysis for molecular toxicity prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ToxicityMetrics:
    """Comprehensive metrics for toxicity prediction evaluation.

    Provides task-specific and aggregated metrics for multi-task
    molecular toxicity prediction models.
    """

    def __init__(self,
                 task_names: List[str],
                 threshold: float = 0.5):
        """Initialize toxicity metrics calculator.

        Args:
            task_names: Names of toxicity prediction tasks
            threshold: Classification threshold for binary metrics
        """
        self.task_names = task_names
        self.threshold = threshold
        self.num_tasks = len(task_names)

    def compute_metrics(self,
                       predictions: np.ndarray,
                       labels: np.ndarray,
                       sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute comprehensive metrics for all tasks.

        Args:
            predictions: Predicted probabilities [n_samples, n_tasks]
            labels: True labels [n_samples, n_tasks]
            sample_weights: Optional sample weights [n_samples]

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Task-specific metrics
        task_aucs = []
        task_aps = []
        task_accs = []
        task_f1s = []

        for i, task_name in enumerate(self.task_names):
            task_preds = predictions[:, i]
            task_labels = labels[:, i]

            # Skip tasks with missing labels
            valid_mask = ~np.isnan(task_labels) & (task_labels != -1)
            if not valid_mask.any():
                continue

            task_preds_valid = task_preds[valid_mask]
            task_labels_valid = task_labels[valid_mask]
            weights = sample_weights[valid_mask] if sample_weights is not None else None

            # AUC-ROC
            try:
                task_auc = roc_auc_score(
                    task_labels_valid,
                    task_preds_valid,
                    sample_weight=weights
                )
                task_aucs.append(task_auc)
                metrics[f'auc_roc_{task_name}'] = task_auc
            except ValueError as e:
                logger.warning(f"Could not compute AUC for {task_name}: {e}")

            # Average Precision
            try:
                task_ap = average_precision_score(
                    task_labels_valid,
                    task_preds_valid,
                    sample_weight=weights
                )
                task_aps.append(task_ap)
                metrics[f'auc_pr_{task_name}'] = task_ap
            except ValueError as e:
                logger.warning(f"Could not compute AP for {task_name}: {e}")

            # Binary classification metrics
            task_binary_preds = (task_preds_valid > self.threshold).astype(int)

            task_acc = accuracy_score(
                task_labels_valid,
                task_binary_preds,
                sample_weight=weights
            )
            task_accs.append(task_acc)
            metrics[f'accuracy_{task_name}'] = task_acc

            task_f1 = f1_score(
                task_labels_valid,
                task_binary_preds,
                sample_weight=weights,
                zero_division=0
            )
            task_f1s.append(task_f1)
            metrics[f'f1_{task_name}'] = task_f1

        # Aggregated metrics
        if task_aucs:
            metrics['auc_roc_mean'] = np.mean(task_aucs)
            metrics['auc_roc_std'] = np.std(task_aucs)
            metrics['auc_roc_median'] = np.median(task_aucs)

        if task_aps:
            metrics['auc_pr_mean'] = np.mean(task_aps)
            metrics['auc_pr_std'] = np.std(task_aps)
            metrics['auc_pr_median'] = np.median(task_aps)

        if task_accs:
            metrics['accuracy_mean'] = np.mean(task_accs)
            metrics['accuracy_std'] = np.std(task_accs)

        if task_f1s:
            metrics['f1_mean'] = np.mean(task_f1s)
            metrics['f1_std'] = np.std(task_f1s)

        return metrics

    def compute_confusion_matrices(self,
                                  predictions: np.ndarray,
                                  labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute confusion matrices for all tasks.

        Args:
            predictions: Predicted probabilities [n_samples, n_tasks]
            labels: True labels [n_samples, n_tasks]

        Returns:
            Dictionary of confusion matrices per task
        """
        confusion_matrices = {}

        for i, task_name in enumerate(self.task_names):
            task_preds = predictions[:, i]
            task_labels = labels[:, i]

            # Skip tasks with missing labels
            valid_mask = ~np.isnan(task_labels) & (task_labels != -1)
            if not valid_mask.any():
                continue

            task_preds_valid = task_preds[valid_mask]
            task_labels_valid = task_labels[valid_mask]

            # Binary predictions
            task_binary_preds = (task_preds_valid > self.threshold).astype(int)

            # Confusion matrix
            cm = confusion_matrix(task_labels_valid, task_binary_preds)
            confusion_matrices[task_name] = cm

        return confusion_matrices

    def plot_roc_curves(self,
                       predictions: np.ndarray,
                       labels: np.ndarray,
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curves for all tasks.

        Args:
            predictions: Predicted probabilities [n_samples, n_tasks]
            labels: True labels [n_samples, n_tasks]
            save_path: Optional path to save plot

        Returns:
            Matplotlib figure
        """
        n_tasks = len(self.task_names)
        cols = min(3, n_tasks)
        rows = (n_tasks + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, task_name in enumerate(self.task_names):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break

            task_preds = predictions[:, i]
            task_labels = labels[:, i]

            # Skip tasks with missing labels
            valid_mask = ~np.isnan(task_labels) & (task_labels != -1)
            if not valid_mask.any():
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{task_name}')
                continue

            task_preds_valid = task_preds[valid_mask]
            task_labels_valid = task_labels[valid_mask]

            try:
                # Compute ROC curve
                fpr, tpr, _ = roc_curve(task_labels_valid, task_preds_valid)
                task_auc = auc(fpr, tpr)

                # Plot
                ax.plot(fpr, tpr, label=f'AUC = {task_auc:.3f}', linewidth=2)
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{task_name}')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)

            except ValueError:
                ax.text(0.5, 0.5, 'Cannot compute ROC', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{task_name}')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_precision_recall_curves(self,
                                    predictions: np.ndarray,
                                    labels: np.ndarray,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot precision-recall curves for all tasks.

        Args:
            predictions: Predicted probabilities [n_samples, n_tasks]
            labels: True labels [n_samples, n_tasks]
            save_path: Optional path to save plot

        Returns:
            Matplotlib figure
        """
        n_tasks = len(self.task_names)
        cols = min(3, n_tasks)
        rows = (n_tasks + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, task_name in enumerate(self.task_names):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break

            task_preds = predictions[:, i]
            task_labels = labels[:, i]

            # Skip tasks with missing labels
            valid_mask = ~np.isnan(task_labels) & (task_labels != -1)
            if not valid_mask.any():
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{task_name}')
                continue

            task_preds_valid = task_preds[valid_mask]
            task_labels_valid = task_labels[valid_mask]

            try:
                # Compute PR curve
                precision, recall, _ = precision_recall_curve(
                    task_labels_valid, task_preds_valid
                )
                ap_score = average_precision_score(
                    task_labels_valid, task_preds_valid
                )

                # Plot
                ax.plot(recall, precision, label=f'AP = {ap_score:.3f}', linewidth=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'{task_name}')
                ax.legend(loc="lower left")
                ax.grid(True, alpha=0.3)

            except ValueError:
                ax.text(0.5, 0.5, 'Cannot compute PR', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{task_name}')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class ScaffoldGeneralizationAnalyzer:
    """Analyzer for scaffold generalization performance.

    Evaluates how well models generalize to unseen molecular scaffolds
    and identifies potential biases in scaffold-based splitting.
    """

    def __init__(self, task_names: List[str]):
        """Initialize scaffold generalization analyzer.

        Args:
            task_names: Names of toxicity prediction tasks
        """
        self.task_names = task_names

    def analyze_scaffold_performance(self,
                                   predictions: np.ndarray,
                                   labels: np.ndarray,
                                   scaffold_ids: List[str],
                                   train_scaffolds: List[str]) -> Dict[str, float]:
        """Analyze performance differences between seen and unseen scaffolds.

        Args:
            predictions: Predicted probabilities [n_samples, n_tasks]
            labels: True labels [n_samples, n_tasks]
            scaffold_ids: Scaffold IDs for each sample
            train_scaffolds: List of scaffolds seen during training

        Returns:
            Dictionary with generalization analysis results
        """
        # Identify seen vs unseen scaffolds
        seen_mask = np.array([scaf in train_scaffolds for scaf in scaffold_ids])
        unseen_mask = ~seen_mask

        if not unseen_mask.any():
            logger.warning("No unseen scaffolds found in evaluation set")
            return {}

        # Compute metrics for seen and unseen scaffolds
        toxicity_metrics = ToxicityMetrics(self.task_names)

        seen_metrics = toxicity_metrics.compute_metrics(
            predictions[seen_mask],
            labels[seen_mask]
        )

        unseen_metrics = toxicity_metrics.compute_metrics(
            predictions[unseen_mask],
            labels[unseen_mask]
        )

        # Compute generalization gaps
        analysis = {
            'seen_scaffold_count': seen_mask.sum(),
            'unseen_scaffold_count': unseen_mask.sum(),
            'seen_auc_roc_mean': seen_metrics.get('auc_roc_mean', 0),
            'unseen_auc_roc_mean': unseen_metrics.get('auc_roc_mean', 0),
            'scaffold_split_generalization_gap': (
                seen_metrics.get('auc_roc_mean', 0) -
                unseen_metrics.get('auc_roc_mean', 0)
            )
        }

        # Task-specific gaps
        for task_name in self.task_names:
            seen_auc = seen_metrics.get(f'auc_roc_{task_name}', 0)
            unseen_auc = unseen_metrics.get(f'auc_roc_{task_name}', 0)
            analysis[f'gap_{task_name}'] = seen_auc - unseen_auc

        return analysis

    def plot_scaffold_performance_comparison(self,
                                           predictions: np.ndarray,
                                           labels: np.ndarray,
                                           scaffold_ids: List[str],
                                           train_scaffolds: List[str],
                                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot performance comparison between seen and unseen scaffolds.

        Args:
            predictions: Predicted probabilities [n_samples, n_tasks]
            labels: True labels [n_samples, n_tasks]
            scaffold_ids: Scaffold IDs for each sample
            train_scaffolds: List of scaffolds seen during training
            save_path: Optional path to save plot

        Returns:
            Matplotlib figure
        """
        # Identify seen vs unseen scaffolds
        seen_mask = np.array([scaf in train_scaffolds for scaf in scaffold_ids])
        unseen_mask = ~seen_mask

        # Compute metrics
        toxicity_metrics = ToxicityMetrics(self.task_names)

        seen_metrics = toxicity_metrics.compute_metrics(
            predictions[seen_mask],
            labels[seen_mask]
        )
        unseen_metrics = toxicity_metrics.compute_metrics(
            predictions[unseen_mask],
            labels[unseen_mask]
        )

        # Prepare data for plotting
        task_aucs_seen = [seen_metrics.get(f'auc_roc_{task}', 0) for task in self.task_names]
        task_aucs_unseen = [unseen_metrics.get(f'auc_roc_{task}', 0) for task in self.task_names]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot comparison
        x = np.arange(len(self.task_names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, task_aucs_seen, width, label='Seen Scaffolds', alpha=0.8)
        bars2 = ax1.bar(x + width/2, task_aucs_unseen, width, label='Unseen Scaffolds', alpha=0.8)

        ax1.set_xlabel('Tasks')
        ax1.set_ylabel('AUC-ROC')
        ax1.set_title('Performance Comparison: Seen vs Unseen Scaffolds')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name[:10] for name in self.task_names], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)

        # Generalization gap plot
        gaps = [task_aucs_seen[i] - task_aucs_unseen[i] for i in range(len(self.task_names))]
        colors = ['red' if gap > 0 else 'green' for gap in gaps]

        bars = ax2.bar(self.task_names, gaps, color=colors, alpha=0.7)
        ax2.set_xlabel('Tasks')
        ax2.set_ylabel('Generalization Gap (Seen - Unseen)')
        ax2.set_title('Scaffold Generalization Gap by Task')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xticklabels([name[:10] for name in self.task_names], rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class MultiTaskEvaluator:
    """Comprehensive evaluator for multi-task molecular property prediction.

    Provides unified evaluation interface with support for multiple metrics
    and statistical significance testing.
    """

    def __init__(self,
                 task_names: List[str],
                 metrics: List[str] = ['auc_roc', 'auc_pr', 'accuracy', 'f1']):
        """Initialize multi-task evaluator.

        Args:
            task_names: Names of prediction tasks
            metrics: List of metrics to compute
        """
        self.task_names = task_names
        self.metrics = metrics
        self.toxicity_metrics = ToxicityMetrics(task_names)

    def evaluate(self,
                 predictions: np.ndarray,
                 labels: np.ndarray,
                 sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate predictions using all configured metrics.

        Args:
            predictions: Predicted probabilities [n_samples, n_tasks]
            labels: True labels [n_samples, n_tasks]
            sample_weights: Optional sample weights [n_samples]

        Returns:
            Dictionary of evaluation results
        """
        return self.toxicity_metrics.compute_metrics(
            predictions, labels, sample_weights
        )

    def bootstrap_confidence_intervals(self,
                                     predictions: np.ndarray,
                                     labels: np.ndarray,
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for metrics.

        Args:
            predictions: Predicted probabilities [n_samples, n_tasks]
            labels: True labels [n_samples, n_tasks]
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary of confidence intervals per metric
        """
        n_samples = predictions.shape[0]
        bootstrap_metrics = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_predictions = predictions[indices]
            boot_labels = labels[indices]

            # Compute metrics
            boot_metrics = self.evaluate(boot_predictions, boot_labels)
            bootstrap_metrics.append(boot_metrics)

        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        confidence_intervals = {}
        for metric in bootstrap_metrics[0].keys():
            values = [m[metric] for m in bootstrap_metrics if metric in m]
            if values:
                lower = np.percentile(values, lower_percentile)
                upper = np.percentile(values, upper_percentile)
                confidence_intervals[metric] = (lower, upper)

        return confidence_intervals

    def compare_models(self,
                      predictions_a: np.ndarray,
                      predictions_b: np.ndarray,
                      labels: np.ndarray,
                      model_names: Tuple[str, str] = ("Model A", "Model B")) -> Dict[str, float]:
        """Compare two models using statistical significance testing.

        Args:
            predictions_a: Predictions from first model
            predictions_b: Predictions from second model
            labels: True labels
            model_names: Names of models being compared

        Returns:
            Dictionary with comparison results
        """
        # Compute metrics for both models
        metrics_a = self.evaluate(predictions_a, labels)
        metrics_b = self.evaluate(predictions_b, labels)

        comparison = {
            'model_a_name': model_names[0],
            'model_b_name': model_names[1],
        }

        # Compare key metrics
        for metric in ['auc_roc_mean', 'auc_pr_mean', 'accuracy_mean', 'f1_mean']:
            if metric in metrics_a and metric in metrics_b:
                a_value = metrics_a[metric]
                b_value = metrics_b[metric]
                diff = a_value - b_value
                pct_diff = (diff / b_value) * 100 if b_value != 0 else 0

                comparison[f'{metric}_diff'] = diff
                comparison[f'{metric}_pct_diff'] = pct_diff
                comparison[f'{metric}_better_model'] = model_names[0] if diff > 0 else model_names[1]

        # Statistical significance testing (McNemar's test for paired predictions)
        try:
            from scipy.stats import chi2

            for i, task_name in enumerate(self.task_names):
                task_labels = labels[:, i]
                valid_mask = ~np.isnan(task_labels) & (task_labels != -1)

                if not valid_mask.any():
                    continue

                task_labels_valid = task_labels[valid_mask]
                preds_a_valid = predictions_a[valid_mask, i] > 0.5
                preds_b_valid = predictions_b[valid_mask, i] > 0.5

                # McNemar's test
                a_correct = (preds_a_valid == task_labels_valid)
                b_correct = (preds_b_valid == task_labels_valid)

                # Contingency table
                both_correct = (a_correct & b_correct).sum()
                a_correct_only = (a_correct & ~b_correct).sum()
                b_correct_only = (~a_correct & b_correct).sum()
                both_wrong = (~a_correct & ~b_correct).sum()

                # McNemar statistic
                if a_correct_only + b_correct_only > 0:
                    mcnemar_stat = (abs(a_correct_only - b_correct_only) - 1)**2 / (a_correct_only + b_correct_only)
                    p_value = 1 - chi2.cdf(mcnemar_stat, 1)
                    comparison[f'mcnemar_pvalue_{task_name}'] = p_value
                    comparison[f'mcnemar_significant_{task_name}'] = p_value < 0.05

        except ImportError:
            logger.warning("scipy not available for significance testing")

        return comparison